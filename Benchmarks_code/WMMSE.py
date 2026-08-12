import os
import copy
import numpy as np
import torch
import json
from pathlib import Path
from utils import (load_channel_data, channel_to_user_coefficients, ri_to_complex_channel, load_llm_model, predict_future_with_llm)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Config
# =========================
nr_of_users = 4
noise_power = 1.0

# WMMSE params
epsilon = 1e-4
max_nr_of_iterations = 100
selected_users = list(range(nr_of_users))

# LLM config
llm_prev_len = 16
llm_pred_len = 4
num_coefficients = 32
llm_patch_size = 4

# =========================
# WMMSE helpers
# =========================
def compute_P(Phi_diag_elements, Sigma_diag_elements, mu):
    mu_array = mu * np.ones(Phi_diag_elements.size)
    result = np.divide(Phi_diag_elements, (Sigma_diag_elements + mu_array) ** 2)
    result = np.sum(result)
    return result

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

def solve_wmmse(
    epsilon,
    channel_input,          # [N, M] complex
    channel_true,          # [N, M] complex
    selected_users,
    total_power,
    noise_power,
    user_weights,
    max_nr_of_iterations,
    power_tolerance=1e-4):
    """
    Run the original WMMSE algorithm to convergence and evaluate the final WSR on channel_true.
    """
    nr_of_users = np.size(channel_input, 0)
    nr_of_BS_antennas = np.size(channel_input, 1)

    break_condition = epsilon + 1.0
    receiver_precoder = np.zeros(nr_of_users, dtype=np.complex128)
    mse_weights = np.ones(nr_of_users, dtype=np.float64)
    transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas), dtype=np.complex128)

    new_receiver_precoder = np.zeros(nr_of_users, dtype=np.complex128)
    new_mse_weights = np.zeros(nr_of_users, dtype=np.float64)
    new_transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas), dtype=np.complex128)

    # Initialization of transmitter precoder using input channel
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            transmitter_precoder[user_index, :] = channel_input[user_index, :]

    norm_init = np.linalg.norm(transmitter_precoder)
    if norm_init > 0:
        transmitter_precoder = transmitter_precoder / norm_init * np.sqrt(total_power)

    nr_of_iteration_counter = 0
    prev_log_term = np.log2(np.prod(mse_weights[selected_users]))
    while break_condition >= epsilon and nr_of_iteration_counter < max_nr_of_iterations:
        nr_of_iteration_counter += 1

        # Step 1: optimize receiver precoder u on input channel
        for user_index_1 in range(nr_of_users):
            if user_index_1 in selected_users:
                user_interference = 0.0
                for user_index_2 in range(nr_of_users):
                    if user_index_2 in selected_users:
                        user_interference += np.abs(np.matmul(np.conj(channel_input[user_index_1, :]),transmitter_precoder[user_index_2, :])) ** 2

                new_receiver_precoder[user_index_1] = (
                    np.matmul(np.conj(channel_input[user_index_1, :]),transmitter_precoder[user_index_1, :])/ (noise_power + user_interference + 1e-12))

        # Step 2: optimize MSE weights w on input channel
        for user_index_1 in range(nr_of_users):
            if user_index_1 in selected_users:
                user_interference = 0.0
                inter_user_interference = 0.0

                for user_index_2 in range(nr_of_users):
                    if user_index_2 in selected_users:
                        user_interference += np.abs(
                            np.matmul(np.conj(channel_input[user_index_1, :]),transmitter_precoder[user_index_2, :])) ** 2

                for user_index_2 in range(nr_of_users):
                    if user_index_2 != user_index_1 and user_index_2 in selected_users:
                        inter_user_interference += np.abs(
                            np.matmul(np.conj(channel_input[user_index_1, :]),transmitter_precoder[user_index_2, :])) ** 2

                new_mse_weights[user_index_1] = ((noise_power + user_interference)/ (noise_power + inter_user_interference + 1e-12))

        # Step 3: optimize transmitter precoder v on predicted channel
        A = np.zeros((nr_of_BS_antennas, nr_of_BS_antennas), dtype=np.complex128)
        for user_index in range(nr_of_users):
            if user_index in selected_users:
                h = np.reshape(channel_input[user_index, :], (nr_of_BS_antennas, 1))
                hh = np.matmul(h, np.conj(h.T))
                A += (new_mse_weights[user_index] * user_weights[user_index] * (np.abs(new_receiver_precoder[user_index]) ** 2) * hh)

        Sigma_diag_elements_true, U = np.linalg.eigh(A)
        Sigma_diag_elements = np.real(Sigma_diag_elements_true)

        Lambda = np.zeros((nr_of_BS_antennas, nr_of_BS_antennas), dtype=np.complex128)
        for user_index in range(nr_of_users):
            if user_index in selected_users:
                h = np.reshape(channel_input[user_index, :], (nr_of_BS_antennas, 1))
                hh = np.matmul(h, np.conj(h.T))
                Lambda += (
                    (user_weights[user_index] ** 2)
                    * (new_mse_weights[user_index] ** 2)
                    * (np.abs(new_receiver_precoder[user_index]) ** 2)
                    * hh
                )

        Phi = np.matmul(np.matmul(np.conj(U.T), Lambda), U)
        Phi_diag_elements = np.real(np.diag(Phi))
        # Bisection search for mu
        mu_low = 0.0
        mu_high = 1.0
        while compute_P(Phi_diag_elements, Sigma_diag_elements, mu_high) > total_power:
            mu_high *= 2.0

        mu_new = (mu_high + mu_low) / 2.0
        obtained_power = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_new)
        max_bisection_iterations = 100
        iteration = 0
        while np.abs(total_power - obtained_power) > power_tolerance and iteration < max_bisection_iterations:
            iteration+=1
            mu_new = (mu_high + mu_low) / 2.0
            obtained_power = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_new)

            if obtained_power > total_power:
                mu_low = mu_new
            else:
                mu_high = mu_new

        mu_star = mu_new
        for user_index in range(nr_of_users):
            if user_index in selected_users:
                inv_term = np.linalg.inv(A + mu_star * np.eye(nr_of_BS_antennas))
                new_transmitter_precoder[user_index, :] = (
                    np.matmul(inv_term, channel_input[user_index, :])
                    * user_weights[user_index]
                    * new_mse_weights[user_index]
                    * new_receiver_precoder[user_index]
                )

        # Break condition on selected users
        new_log_term = np.log2(np.prod(new_mse_weights[selected_users]))
        break_condition = np.abs(new_log_term - prev_log_term)
        prev_log_term = new_log_term
        mse_weights = copy.deepcopy(new_mse_weights)
        transmitter_precoder = copy.deepcopy(new_transmitter_precoder)
        receiver_precoder = copy.deepcopy(new_receiver_precoder)
    # IMPORTANT: final WSR is evaluated on TRUE channel
    final_wsr_true = compute_weighted_sum_rate(user_weights, channel_true, transmitter_precoder, noise_power, selected_users)
    return transmitter_precoder, receiver_precoder, mse_weights, final_wsr_true

# =========================
# Benchmark evaluation
# =========================
def evaluate_wmmse_benchmark(
    input_userwise,      # [S,N,fut_len,M,2]
    true_userwise,      # [S,N,fut_len,M,2]
    total_power,
    noise_power,
    mode,
    epsilon=1e-4,
    max_nr_of_iterations=100,
    selected_users=None,
    delta=1):

    """
    input_userwise:
        current   -> [S,N,hist_len,M,2]
        genie     -> [S,N,fut_len,M,2]
        predicted -> [S,N,fut_len,M,2]

    true_userwise:
        [S,N,fut_len,M,2]
    """
    if mode not in ("current", "genie", "predicted"):
        raise ValueError("mode must be one of: 'current', 'genie', 'predicted'")

    if mode != "current" and delta > input_userwise.shape[2]:
        raise ValueError(f"delta={delta} is outside the input future horizon")

    if selected_users is None:
        selected_users = list(range(input_userwise.shape[1]))

    # make sure everything is NumPy before entering original WMMSE code
    if torch.is_tensor(input_userwise):
        input_userwise = input_userwise.detach().cpu().numpy()
    if torch.is_tensor(true_userwise):
        true_userwise = true_userwise.detach().cpu().numpy()

    S = input_userwise.shape[0]
    user_weights = np.ones(input_userwise.shape[1], dtype=np.float64)

    wsr_list = []

    for s in range(S):
        # CSI used to design the beamformer
        if mode == "current":
            input_channel = input_userwise[s, :, -1, :, :]  # H_hat_n [N,M,2]

        else:  # genie or predicted
            input_channel = input_userwise[s, :, delta - 1, :, :]  # H_{n+delta} or H_hat_{n+delta} [N,M,2]

        # True future channel used for evaluation in all modes
        true_channel = true_userwise[s, :, delta - 1, :, :]  # H_{n+delta}

        input_channel_complex = ri_to_complex_channel(input_channel)   # [N,M]
        true_channel_complex = ri_to_complex_channel(true_channel)    # [N,M]

        _, _, _, final_wsr_true = solve_wmmse(epsilon=epsilon,channel_input=input_channel_complex ,
            channel_true=true_channel_complex,selected_users=selected_users,total_power=total_power,noise_power=noise_power,
            user_weights=user_weights,max_nr_of_iterations=max_nr_of_iterations,power_tolerance=1e-4)

        wsr_list.append(final_wsr_true)

    wsr_array = np.array(wsr_list, dtype=np.float64)
    return wsr_array, np.mean(wsr_array), np.std(wsr_array)

# =========================
# Main
# =========================
if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)

    scenario = "UMa"  # "UMa" or "RMa"
    mode = "predicted"  # "current", "genie", or "predicted"
    delta = 1
    print(f"Scenario: {scenario} | Mode: {mode}")

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    hist_path = PROJECT_ROOT / "data" / scenario / "test" / "H_hist_test_noisy.mat"
    fut_path = PROJECT_ROOT / "data" / scenario / "test" / "H_fut_test.mat"
    hist_key = "H_hist_test"
    fut_key = "H_fut_test"
    norm_stats_path = (PROJECT_ROOT / "Weights" / f"LLM4CP_{scenario}_norm_stats.json")
    model_path = (PROJECT_ROOT / "Weights" / f"LLM4CP_{scenario}.pth")
    benchmark_snr_save_path = (PROJECT_ROOT / "results" / f"benchmark_test_results_wsr_vs_snr_{scenario}_{mode}_delta{delta}.npz")

    # -------- load raw data --------
    H_hist, H_fut = load_channel_data(hist_path,fut_path,hist_key=hist_key,fut_key=fut_key)
    print("Raw hist shape:", H_hist.shape)
    print("Raw fut  shape:", H_fut.shape)

    if mode == "predicted":
        # -------- load normalization stats --------
        with open(norm_stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        norm_stats = {"mean": torch.tensor(stats["mean"], dtype=torch.float32), "std": torch.tensor(stats["std"], dtype=torch.float32)}

        # -------- load pretrained LLM --------
        assert os.path.exists(model_path), f"Missing LLM weights at {model_path}"
        llm_model = load_llm_model(model_path)
        print("Loaded LLM weights.")

        # -------- predict future channels with LLM --------
        predicted_future_channel = predict_future_with_llm(model=llm_model,H_hist_raw=H_hist,norm_stats=norm_stats,batch_size=512,num_coefficients=num_coefficients)   # [S,U,L,32,2], ORIGINAL SCALE

    # -------- convert true future to userwise real-imag --------
    future_clean_channel = channel_to_user_coefficients(H_fut)   # [S,N,fut_len,32,2]
    current_noisy_channel = channel_to_user_coefficients(H_hist) # [S,N,hist_len,32,2]
    print("current noisy channel shape:", current_noisy_channel.shape)
    print("future clean channel shape:", future_clean_channel.shape)

    if mode == "current":
        wmmse_input = current_noisy_channel

    elif mode == "genie":
        wmmse_input = future_clean_channel

    elif mode == "predicted":
        wmmse_input = predicted_future_channel

    else:
        raise ValueError("mode must be one of: 'current', 'genie', 'predicted'")

    SNR_range = [0, 2.5, 5, 7.5, 10, 12.5]
    benchmark_mean_vs_snr = []
    benchmark_std_vs_snr = []
    for snr_db in SNR_range:
        total_power_snr = 10 ** (snr_db / 10)
        print(f"\n========== Evaluating Original WMMSE at SNR={snr_db} dB ==========")
        wsr_array, mean_wsr_test, std_wsr_test = evaluate_wmmse_benchmark(input_userwise=wmmse_input,true_userwise=future_clean_channel,
            total_power=total_power_snr,noise_power=float(noise_power),mode=mode,epsilon=epsilon,max_nr_of_iterations=max_nr_of_iterations, selected_users=selected_users,delta=delta)

        benchmark_mean_vs_snr.append(mean_wsr_test)
        benchmark_std_vs_snr.append(std_wsr_test)
        print(f"SNR={snr_db} dB | Mean WSR={mean_wsr_test:.6f} | Std={std_wsr_test:.6f}")

    os.makedirs("results", exist_ok=True)
    np.savez(benchmark_snr_save_path,snr_db=np.array(SNR_range),mean_wsr=np.array(benchmark_mean_vs_snr),std_wsr=np.array(benchmark_std_vs_snr))
    print(f"Saved benchmark SNR results to: {benchmark_snr_save_path}")
