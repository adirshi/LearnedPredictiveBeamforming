import os
import torch
import json
import time
import hdf5storage
import numpy as np
from pathlib import Path
from einops import rearrange
from kalman_filter import predict_future_with_kf
from utils import (load_channel_data, channel_to_user_coefficients, ri_to_complex_channel, load_llm_model,
                   channel_history_to_coefficient_sequences,normalize_tensor,denormalize_tensor,coefficient_sequences_to_user_channels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Config
# =========================
N = 4
noise_power = 1.0

# WMMSE params
epsilon = 1e-4
max_nr_of_iterations = 100
selected_users = list(range(N))

# LLM config
num_coefficients = 32

# =========================
# WMMSE helpers
# =========================
def compute_p(Phi_diag_elements,Sigma_diag_elements,mu):
    return torch.sum( Phi_diag_elements / (Sigma_diag_elements + mu) ** 2)

def compute_sinr(channel, precoder, noise_power, user_index, selected_users):
    """
    channel  : [N, M] complex
    precoder : [N, M] complex
    """
    signal = np.abs(np.matmul(np.conj(channel[user_index, :]), precoder[user_index, :])) ** 2
    interference = 0.0
    for j in range(channel.shape[0]):
        if j != user_index and j in selected_users:
            interference += np.abs(np.matmul(np.conj(channel[user_index, :]), precoder[j, :])) ** 2
    return signal / (noise_power + interference + 1e-12)

def compute_weighted_sum_rate(user_weights, channel, precoder, noise_power, selected_users):
    result = 0.0
    nr_of_users = np.size(channel, 0)
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            user_sinr = compute_sinr(channel, precoder, noise_power, user_index, selected_users)
            result += user_weights[user_index] * np.log2(1.0 + user_sinr)
    return result

@torch.no_grad()
def run_wmmse(epsilon,channel_input, selected_users, total_power, noise_power, user_weights, max_nr_of_iterations, power_tolerance=1e-4):
    """
    channel_input: [N,M] complex
    Returns:
        transmitter_precoder : [N,M] complex128
        receiver_precoder    : [N]   complex128
        mse_weights          : [N]   float64
    """

    # ---------------------------------------------------------
    # Convert input to GPU tensors
    # ---------------------------------------------------------
    N, M = channel_input.shape
    selected_users_tensor = torch.as_tensor(selected_users,dtype=torch.long,device=device)

    # ---------------------------------------------------------
    # Extract selected users
    # ---------------------------------------------------------
    H = channel_input[selected_users_tensor]       # [Ns,M]
    alpha = user_weights[selected_users_tensor]    # [Ns]
    Ns = H.shape[0]

    # ---------------------------------------------------------
    # Scalars as GPU tensors
    # ---------------------------------------------------------
    total_power_t = torch.as_tensor(total_power,dtype=torch.float64,device=device)
    noise_power_t = torch.as_tensor(noise_power,dtype=torch.float64,device=device)

    # ---------------------------------------------------------
    # Beamformer initialization:
    # ---------------------------------------------------------
    V = H.clone()                                  # [Ns,M]

    norm_init = torch.linalg.norm(V)

    if norm_init.item() > 0:
        V = (V / norm_init) * torch.sqrt(total_power_t)

    # Initialization
    u = torch.zeros(Ns,dtype=torch.complex128,device=device)
    w = torch.ones(Ns,dtype=torch.float64,device=device)

    # Identity used in V update
    eye_M = torch.eye(M,dtype=torch.complex128,device=device)

    # ---------------------------------------------------------
    # Stopping-rule initialization
    # ---------------------------------------------------------
    break_condition = epsilon + 1.0
    prev_log_term = torch.log2(torch.prod(w)).item()
    nr_of_iteration_counter = 0

    # =========================================================
    # Main WMMSE loop
    # =========================================================
    while break_condition >= epsilon and nr_of_iteration_counter < max_nr_of_iterations:
        nr_of_iteration_counter += 1
        # =====================================================
        # Compute all h_i^H v_j
        # G[i,j] = h_i^H v_j
        # =====================================================
        G = H.conj() @ V.T                         # [Ns,Ns]
        power_matrix = torch.abs(G) ** 2            # [Ns,Ns]

        # sum_j |h_i^H v_j|^2
        user_interference = torch.sum(power_matrix,dim=1) # [Ns]

        # h_i^H v_i
        desired_signal = torch.diagonal(G)                # [Ns]

        # |h_i^H v_i|^2
        desired_power = torch.abs(desired_signal) ** 2    # [Ns]

        # =====================================================
        # Step 1: receiver precoder
        # u_i = h_i^H v_i / (noise + sum_j |h_i^H v_j|^2)
        # =====================================================
        new_u = desired_signal / (noise_power_t + user_interference + 1e-12)  # [Ns]

        # =====================================================
        # Step 2: MSE weights
        # inter_user_interference = sum_{j != i}|h_i^H v_j|^2
        # =====================================================
        inter_user_interference = user_interference- desired_power
        new_w = (noise_power_t + user_interference) / (noise_power_t + inter_user_interference + 1e-12) # [Ns]

        # =====================================================
        # Step 3: construct A
        # A = sum_i w_i alpha_i |u_i|^2 h_i h_i^H
        # =====================================================
        coeff_A = (new_w * alpha * torch.abs(new_u) ** 2)    # [Ns]
        A = H.T @ (coeff_A[:, None] * H.conj())              # [M,M]

        # =====================================================
        # Eigen-decomposition
        # =====================================================
        Sigma_diag_elements, U = torch.linalg.eigh(A) # [M]

        # =====================================================
        # Construct Lambda
        # Lambda = sum_i alpha_i^2 w_i^2 |u_i|^2 h_i h_i^H
        # =====================================================
        coeff_Lambda = alpha ** 2 * new_w ** 2 * torch.abs(new_u) ** 2
        Lambda = H.T @ (coeff_Lambda[:, None] * H.conj())  # [M,M]

        # =====================================================
        # Phi = U^H Lambda U
        # =====================================================
        Phi = U.conj().T @ Lambda @ U
        Phi_diag_elements = torch.diagonal(Phi).real # [M]

        # =====================================================
        # Bisection search for mu
        # =====================================================
        mu_low = 0.0
        mu_high = 1.0

        while compute_p(Phi_diag_elements,Sigma_diag_elements,mu_high).item() > total_power:
            mu_high *= 2.0

        mu_new = (mu_high + mu_low) / 2.0
        obtained_power = compute_p(Phi_diag_elements,Sigma_diag_elements,mu_new).item()
        max_bisection_iterations = 100
        iteration = 0

        while abs(total_power - obtained_power) > power_tolerance and iteration < max_bisection_iterations:
            iteration += 1
            mu_new = (mu_high + mu_low) / 2.0
            obtained_power = compute_p(Phi_diag_elements,Sigma_diag_elements,mu_new).item()
            if obtained_power > total_power:
                mu_low = mu_new
            else:
                mu_high = mu_new

        mu_star = mu_new
        # =====================================================
        # inv_term = inv(A + mu I)
        # =====================================================
        inv_term = torch.linalg.inv(A + mu_star * eye_M)   # [M,M]

        # -----------------------------------------------------
        # v_i = inv_term @ h_i * alpha_i * w_i * u_i
        # Vectorized version: H.T = [M,Ns]
        # inv_term @ H.T = [M,Ns]
        # -----------------------------------------------------
        new_V = (inv_term @ H.T).T        # [Ns,M]
        scaling = alpha * new_w * new_u   # [Ns]
        new_V = new_V * scaling[:, None]  # [Ns,M]

        # =====================================================
        # Same break condition
        # =====================================================
        new_log_term = torch.log2(torch.prod(new_w)).item()
        break_condition = abs(new_log_term - prev_log_term)
        prev_log_term = new_log_term

        w = new_w.clone()
        V = new_V.clone()
        u = new_u.clone()

    transmitter_precoder = torch.zeros((N, M),dtype=torch.complex128,device=device)
    receiver_precoder = torch.zeros(N,dtype=torch.complex128,device=device)
    mse_weights = torch.zeros(N,dtype=torch.float64,device=device)
    transmitter_precoder[selected_users_tensor] = V
    receiver_precoder[selected_users_tensor] = u
    mse_weights[selected_users_tensor] = w

    return transmitter_precoder, receiver_precoder, mse_weights

# =========================
# Benchmark evaluation
# =========================
def evaluate_wmmse(input_userwise, true_userwise,
    total_power, noise_power, mode, epsilon=1e-4, max_nr_of_iterations=100, selected_users=None, delta=1):
    """
    input_userwise: current [S,N,hist_len,M,2] or genie [S,N,fut_len,M,2]
    true_userwise: [S,N,fut_len,M,2]
    """

    if mode not in ("current", "genie"):
        raise ValueError("mode must be one of: 'current', 'genie'")

    if mode != "current" and delta > input_userwise.shape[2]:
        raise ValueError(f"delta={delta} is outside the input future horizon")

    if selected_users is None:
        selected_users = list(range(input_userwise.shape[1]))

    S = input_userwise.shape[0]
    user_weights_gpu = torch.ones(N,dtype=torch.float64,device=device)
    user_weights_np = np.ones(N,dtype=np.float64)

    wsr_list = []

    for s in range(S):
        # CSI used to design the beamformer
        if mode == "current":
            input_channel = input_userwise[s, :, -1, :, :]  # H_n [N,M,2]

        else:
            input_channel = input_userwise[s, :, delta - 1, :, :]  # H_{n+delta} [N,M,2]

        # True future channel used for evaluation in all modes
        true_channel = true_userwise[s, :, delta - 1, :, :]  # H_{n+delta}
        input_channel_complex = torch.as_tensor(ri_to_complex_channel(input_channel),dtype=torch.complex128,device=device)  # [N,M]
        if torch.is_tensor(true_channel):
            true_channel = true_channel.detach().cpu().numpy()

        true_channel_complex = ri_to_complex_channel(true_channel)

        V, _, _ = run_wmmse(epsilon=epsilon,channel_input=input_channel_complex , selected_users=selected_users,total_power=total_power,noise_power=noise_power,
            user_weights=user_weights_gpu,max_nr_of_iterations=max_nr_of_iterations,power_tolerance=1e-4)
        V_np = V.detach().cpu().numpy()
        final_wsr_true = compute_weighted_sum_rate(user_weights_np,true_channel_complex,V_np,noise_power,selected_users)
        wsr_list.append(final_wsr_true)

    wsr_array = np.array(wsr_list, dtype=np.float64)
    return wsr_array, np.mean(wsr_array), np.std(wsr_array)

@torch.no_grad()
def evaluate_llm4cp_plus_wmmse(model, H_hist_raw, true_userwise, norm_stats, total_power, noise_power, epsilon=1e-4,
        max_nr_of_iterations=100, selected_users=None, delta=1, num_coefficients=32, warmup_samples=10):

    # ---------------------------------------------------------
    # [S,N,hist_len,Pol,Mv,Mh] -> [S*N*M,P,2]
    # ---------------------------------------------------------
    x_llm, S, N = channel_history_to_coefficient_sequences(H_hist_raw, num_coefficients=num_coefficients)
    x_llm_norm = normalize_tensor(x_llm, norm_stats)
    sequences_per_sample = N * num_coefficients   # 4*32 = 128

    if selected_users is None:
        selected_users = list(range(N))

    user_weights_gpu = torch.ones(N,dtype=torch.float64,device=device)
    user_weights_np = np.ones(N,dtype=np.float64)

    wsr_list = []
    latency_list = []
    prediction_latency_list = []
    for s in range(S):
        start_idx = s * sequences_per_sample
        end_idx = (s + 1) * sequences_per_sample
        x_sample = x_llm_norm[start_idx:end_idx].to(device)  # [128,hist_len,2]

        measure_time = s >= warmup_samples
        if measure_time:
            torch.cuda.synchronize()
            start_time = time.perf_counter()

        # =====================================================
        # 1. LLM4CP channel prediction
        # =====================================================
        y_sample = model(x_sample,None,None,None)  # [128,fut_len,2]

        # External de-normalization
        y_sample = denormalize_tensor(y_sample, norm_stats)

        # -----------------------------------------------------
        # Reconstruct ONE complete predicted channel
        # [128,fut_len,2] -> [1,N,fut_len,M,2]
        # -----------------------------------------------------
        pred_userwise = coefficient_sequences_to_user_channels(y_sample,S=1,N=N,num_coefficients=num_coefficients)
        pred_channel = pred_userwise[0, :, delta - 1, :, :]  # [N,M,2]
        input_channel_complex = ri_to_complex_channel(pred_channel).to(torch.complex128)  # [N,M]

        if measure_time:
            torch.cuda.synchronize()
            pred_time = time.perf_counter()

        # =====================================================
        # 2. WMMSE on predicted channel
        # =====================================================
        V, _, _ = run_wmmse(epsilon=epsilon,channel_input=input_channel_complex,selected_users=selected_users,total_power=total_power,noise_power=noise_power,user_weights=user_weights_gpu,
            max_nr_of_iterations=max_nr_of_iterations, power_tolerance=1e-4)

        # =====================================================
        # End-to-end latency:
        # LLM prediction + WMMSE
        # =====================================================
        if measure_time:
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            elapsed_ms = (end_time - start_time) * 1000.0
            prediction_latency = (pred_time - start_time) * 1000.0
            latency_list.append(elapsed_ms)
            prediction_latency_list.append(prediction_latency)


        true_channel = true_userwise[s, :, delta - 1, :, :]  # [N,M,2]
        if torch.is_tensor(true_channel):
            true_channel = true_channel.detach().cpu().numpy()

        true_channel_complex = ri_to_complex_channel(true_channel)
        V_np = V.detach().cpu().numpy()
        final_wsr_true = compute_weighted_sum_rate(user_weights_np,true_channel_complex,V_np,noise_power,selected_users)
        wsr_list.append(final_wsr_true)

    wsr_array = np.asarray(wsr_list)
    return np.mean(wsr_array), np.std(wsr_array), np.mean(latency_list) , np.mean(prediction_latency_list)

def evaluate_kf_plus_wmmse(H_hist_raw, true_userwise, measurement_noise_var, ue_speed,
        total_power, noise_power, epsilon=1e-4, max_nr_of_iterations=100, selected_users=None, delta=1, T=5, num_coefficients=32,
        carrier_frequency=2.4e9, sampling_interval=0.5e-3, channel_variance=1.0, warmup_samples=10):

    # Keep last T historical samples
    H_hist_seq = H_hist_raw[:, :, -T:, :, :, :] # [S,N,T,Pol,Mv,Mh]
    H_hist_seq = torch.as_tensor(H_hist_seq, dtype=torch.complex64, device=device)
    measurement_noise_var = torch.as_tensor(measurement_noise_var, dtype=torch.float32, device=device)
    ue_speed = torch.as_tensor(ue_speed, dtype=torch.float32, device=device)

    # Flatten channel coefficients
    H_hist_seq = rearrange(H_hist_seq,'s n t pol mv mh -> s n t (pol mv mh)')# [S,N,T,M]
    S, N, _, M = H_hist_seq.shape

    assert M == num_coefficients
    assert measurement_noise_var.shape == (S, N)
    assert ue_speed.shape == (S, N)
    if selected_users is None:
        selected_users = list(range(N))

    user_weights_gpu = torch.ones(N,dtype=torch.float64,device=device)
    user_weights_np = np.ones(N,dtype=np.float64)

    wsr_list = []
    latency_list = []
    prediction_latency_list = []
    # =========================================================
    # One complete realization at a time (for the latency computation)
    # =========================================================
    for s in range(S):
        # Inputs of scene s are prepared BEFORE timing
        H_sample = H_hist_seq[s] # [N,T,M]
        R_sample = measurement_noise_var[s] # [N]
        speed_sample = ue_speed[s]# [N]
        measure_time = s >= warmup_samples

        # =====================================================
        # Start end-to-end timer
        # =====================================================
        if measure_time:
            torch.cuda.synchronize()
            start_time = time.perf_counter()

        # =====================================================
        # 1. KF CHANNEL PREDICTION
        # =====================================================
        predicted_channel_complex = predict_future_with_kf( H_hist_seq=H_sample, measurement_noise_var=R_sample, ue_speed=speed_sample, delta=delta,
                carrier_frequency=carrier_frequency, sampling_interval=sampling_interval, channel_variance=channel_variance).to(torch.complex128) # [N,M] complex

        # -----------------------------------------------------
        # Prediction latency ends HERE
        # -----------------------------------------------------
        if measure_time:
            torch.cuda.synchronize()
            pred_time = time.perf_counter()

        # =====================================================
        # 2. WMMSE
        # =====================================================
        V, _, _ = run_wmmse(epsilon=epsilon,channel_input=predicted_channel_complex,
            selected_users=selected_users,total_power=total_power,noise_power=noise_power,user_weights=user_weights_gpu,max_nr_of_iterations=max_nr_of_iterations,power_tolerance=1e-4)

        # =====================================================
        # End end-to-end latency
        # =====================================================
        if measure_time:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            prediction_latency_ms = (pred_time - start_time) * 1000.0
            end_to_end_latency_ms = (end_time - start_time) * 1000.0
            prediction_latency_list.append(prediction_latency_ms)
            latency_list.append(end_to_end_latency_ms)

        true_channel = true_userwise[s, :, delta - 1, :, :]# [N,M,2]
        if torch.is_tensor(true_channel):
            true_channel = true_channel.detach().cpu().numpy()

        true_channel_complex = ri_to_complex_channel(true_channel)
        V_np = V.detach().cpu().numpy()
        final_wsr_true = compute_weighted_sum_rate(user_weights_np,true_channel_complex,V_np,noise_power,selected_users)
        wsr_list.append(final_wsr_true)

    wsr_array = np.asarray(wsr_list,dtype=np.float64)
    return np.mean(wsr_array), np.std(wsr_array), np.mean(latency_list), np.mean(prediction_latency_list)

# =========================
# Main
# =========================
if __name__ == "__main__":

    torch.manual_seed(1234)
    np.random.seed(1234)

    scenario = "RMa"  # "UMa" or "RMa"
    mode = "KF"   # "current", "genie", "LLM4CP" or "KF"
    delta = 1

    SNR_range = [0, 2.5, 5, 7.5, 10, 12.5]

    print(f"Scenario: {scenario} | Mode: {mode}")

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    hist_path = PROJECT_ROOT / "data" / scenario / "test" / "H_hist_test_noisy.mat"
    fut_path = PROJECT_ROOT / "data" / scenario / "test" / "H_fut_test.mat"
    UE_speed_path = PROJECT_ROOT / "data" / scenario / "test" / "UEspeed_test.mat"

    hist_key = "H_hist_test"
    fut_key = "H_fut_test"

    norm_stats_path = PROJECT_ROOT / "Weights" / f"LLM4CP_{scenario}_norm_stats.json"
    model_path = PROJECT_ROOT / "Weights" / f"LLM4CP_{scenario}.pth"

    benchmark_save_path = PROJECT_ROOT / "results"/ f"benchmark_test_results_wsr_vs_snr_{scenario}_{mode}_delta{delta}.npz"

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    H_hist, H_fut = load_channel_data(hist_path,fut_path,hist_key=hist_key,fut_key=fut_key)

    print("Raw hist shape:", H_hist.shape)
    print("Raw fut  shape:", H_fut.shape)

    future_clean_channel = channel_to_user_coefficients(H_fut)
    current_noisy_channel = channel_to_user_coefficients(H_hist)

    print("current noisy channel shape:", current_noisy_channel.shape)
    print("future clean channel shape:", future_clean_channel.shape)

    benchmark_mean_vs_snr = []
    benchmark_std_vs_snr = []
    latency_vs_snr = []
    prediction_latency_vs_snr = []

    # =========================================================
    # LLM4CP + WMMSE
    # =========================================================
    if mode == "LLM4CP":
        with open(norm_stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        norm_stats = {"scale": torch.tensor(stats["scale"], dtype=torch.float32)}
        assert os.path.exists(model_path), f"Missing LLM weights at {model_path}"
        llm_model = load_llm_model(model_path)
        llm_model.eval()
        print("Loaded LLM weights.")

        for snr_db in SNR_range:
            total_power_snr = 10 ** (snr_db / 10)
            print(f"\n========== LLM4CP + WMMSE | SNR={snr_db} dB ==========")
            mean_wsr_test, std_wsr_test, avg_latency_ms , avg_pred_latency = evaluate_llm4cp_plus_wmmse(model=llm_model,
                    H_hist_raw=H_hist,true_userwise=future_clean_channel,norm_stats=norm_stats,total_power=total_power_snr,noise_power=float(noise_power),epsilon=epsilon,
                    max_nr_of_iterations=max_nr_of_iterations,selected_users=selected_users,delta=delta,num_coefficients=num_coefficients,warmup_samples=10)

            benchmark_mean_vs_snr.append(mean_wsr_test)
            benchmark_std_vs_snr.append(std_wsr_test)
            latency_vs_snr.append(avg_latency_ms)
            prediction_latency_vs_snr.append(avg_pred_latency)
            print(f"SNR={snr_db} dB | Mean WSR={mean_wsr_test:.6f} | Std={std_wsr_test:.6f} | Prediction latency={avg_pred_latency:.4f} ms, End-to-end latency={avg_latency_ms:.4f} ms")
        print(f"Avg prediction latency: {np.mean(prediction_latency_vs_snr):.4f} ms | Avg latency: {np.mean(latency_vs_snr):.4f} ms")

    # =========================================================
    # KF + WMMSE
    # =========================================================
    elif mode == "KF":

        measurement_noise_var = hdf5storage.loadmat(hist_path)["measurement_noise_var"]
        ue_speed = hdf5storage.loadmat(UE_speed_path)["UEspeed_test"]

        for snr_db in SNR_range:
            total_power_snr = 10 ** (snr_db / 10)
            print(f"\n========== Velocity-Aware KF + WMMSE | SNR={snr_db} dB ==========")

            mean_wsr_test,std_wsr_test,avg_latency_ms,avg_pred_latency = evaluate_kf_plus_wmmse(H_hist_raw=H_hist,true_userwise=future_clean_channel,measurement_noise_var=measurement_noise_var,
                ue_speed=ue_speed,total_power=total_power_snr,noise_power=float(noise_power),epsilon=epsilon,max_nr_of_iterations=max_nr_of_iterations,selected_users=selected_users,
                delta=delta,T=5,num_coefficients=num_coefficients, warmup_samples=10)

            benchmark_mean_vs_snr.append(mean_wsr_test)
            benchmark_std_vs_snr.append(std_wsr_test)
            latency_vs_snr.append(avg_latency_ms)
            prediction_latency_vs_snr.append(avg_pred_latency)
            print(f"SNR={snr_db} dB | Mean WSR={mean_wsr_test:.6f} | Std={std_wsr_test:.6f} "
                  f"Prediction latency={avg_pred_latency:.4f} ms | End-to-end latency={avg_latency_ms:.4f} ms")

        print(f"Avg prediction latency: {np.mean(prediction_latency_vs_snr):.4f} ms | "
              f"Avg latency: {np.mean(latency_vs_snr):.4f} ms")

    # =========================================================
    # Current CSI / Genie CSI + WMMSE
    # =========================================================
    elif mode in ("current", "genie"):

        if mode == "current":
            wmmse_input = current_noisy_channel

        else:  # genie
            wmmse_input = future_clean_channel

        for snr_db in SNR_range:
            total_power_snr = 10 ** (snr_db / 10)
            print(f"\n========== Evaluating WMMSE at SNR={snr_db} dB ==========")

            wsr_array, mean_wsr_test, std_wsr_test = evaluate_wmmse(input_userwise=wmmse_input,true_userwise=future_clean_channel,total_power=total_power_snr,
                                                                              noise_power=float(noise_power),mode=mode,epsilon=epsilon,max_nr_of_iterations=max_nr_of_iterations,
                                                                              selected_users=selected_users, delta=delta)

            benchmark_mean_vs_snr.append(mean_wsr_test)
            benchmark_std_vs_snr.append(std_wsr_test)
            print(f"SNR={snr_db} dB | Mean WSR={mean_wsr_test:.6f} | Std={std_wsr_test:.6f}")

    else:
        raise ValueError("mode must be one of: 'current', 'genie', 'LLM4CP', 'KF'")

    # =========================================================
    # Save results
    # =========================================================
    os.makedirs(PROJECT_ROOT / "results", exist_ok=True)
    save_dict = {"snr_db": np.array(SNR_range), "mean_wsr": np.array(benchmark_mean_vs_snr), "std_wsr": np.array(benchmark_std_vs_snr)}
    if mode in ("LLM4CP", "KF"):
        save_dict["latency_ms"] = np.array(latency_vs_snr)
        save_dict["prediction_latency_ms"] = np.array(prediction_latency_vs_snr)

    np.savez(benchmark_save_path, **save_dict)
    print(f"Saved benchmark results to: {benchmark_save_path}")