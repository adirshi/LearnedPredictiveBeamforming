import numpy as np
import torch
import matplotlib.pyplot as plt
from metrics import compute_WSR
from utils import load_channel_data, prepare_wmmse_channels, load_mobility_norm_stats, normalize_mobility_test, load_posvel, build_mobility_npgr_input, build_cache
from predictive_unfolded_wmmse import Predictive_UnfoldedWMMSE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
DTYPE = torch.float32

# =========================
# Config
# =========================
K = 4
L = 5
T = 5
delta = 1
SNR_range = [0, 2.5, 5, 7.5, 10, 12.5]
noise_power = torch.tensor(1.0, dtype=DTYPE, device=device)
scenario = "UMa"  # "UMa" or "RMa"

# test files
hist_path = f"data/{scenario}/test/H_hist_test_noisy.mat"
fut_path = f"data/{scenario}/test/H_fut_test.mat"

# keys inside mat files
hist_key = "H_hist_test"
fut_key = "H_fut_test"

Pxy_test_path = f"data/{scenario}/test/Pxy_hist_test.mat"
Vxy_test_path = f"data/{scenario}/test/Vxy_hist_test.mat"
pxy_key = "Pxy_test"
vxy_key = "Vxy_test"

# Benchmark result files
benchmark_LLM4CP = f"results/benchmark_test_results_wsr_vs_snr_{scenario}_LLM4CP_delta{delta}.npz"
benchmark_KF = f"results/benchmark_test_results_wsr_vs_snr_{scenario}_KF_delta{delta}.npz"
benchmark_genie = f"results/benchmark_test_results_wsr_vs_snr_{scenario}_genie_delta{delta}.npz"
benchmark_cur_csi = f"results/benchmark_test_results_wsr_vs_snr_{scenario}_current_delta{delta}.npz"

# Model checkpoint
model_path = f"Weights/Predictive_unfolded_WMMSE_model_{scenario}_delta{delta}_L{L}.pt"

@torch.no_grad()
def evaluate_vs_snr_predictive_model(channel_tplusd,channel_hist_seq,mob_features,checkpoint_path,K,delta=1,batch_size=64):
    if device.type != "cuda":
        raise RuntimeError("Latency measurement requires a CUDA-enabled GPU.")

    S = channel_hist_seq.shape[0]
    _, N, _, _ = channel_tplusd.shape
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    L = int(checkpoint["L"])
    Gamma = checkpoint["Gamma"].to(device=device, dtype=DTYPE)
    model =  Predictive_UnfoldedWMMSE(L, N,K,noise_power, Gamma, delta).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    user_weights = torch.ones((1, N, 1), device=device, dtype=DTYPE)

    model.eval()
    results = []
    latency_results = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for snr_db in SNR_range:
        total_power = 10 ** (snr_db / 10)
        weighted_wsr_sum = 0.0
        total_samples = 0
        timed_samples = 0
        total_inference_time = 0.0
        for start in range(0, S, batch_size):
            end = min(start + batch_size, S)
            ch_tplusd_batch = channel_tplusd[start:end]
            H_hist_seq_batch = channel_hist_seq[start:end]
            mob_batch = mob_features[start:end]

            H_past = H_hist_seq_batch[:, :-1]  # t-T+1, ..., t-1
            H_t = H_hist_seq_batch[:, -1]  # t
            B = ch_tplusd_batch.shape[0]
            cache = build_cache(model=model, H_past=H_past, user_weights=user_weights, total_power=total_power)

            # Skip first 20 runs for GPU warm-up
            measure_time = total_samples >= 20
            if measure_time:
                torch.cuda.synchronize()
                starter.record()

            # Inference
            V_final, _ = model(H_t, mob_batch, cache, total_power, user_weights, stage=2)
            if measure_time:
                ender.record()
                torch.cuda.synchronize()
                elapsed_ms = starter.elapsed_time(ender)
                total_inference_time += elapsed_ms
                timed_samples += B
            WSR = compute_WSR(noise_power, user_weights, ch_tplusd_batch, V_final)

            weighted_wsr_sum += WSR.item() * B
            total_samples += B

        avg_wsr = weighted_wsr_sum / total_samples
        results.append(avg_wsr)
        latency_results.append(total_inference_time / timed_samples)
        print(f"[Predictive] SNR={snr_db} dB| WSR={avg_wsr:.6f} ")
        print(f"Average inference time: {total_inference_time / timed_samples:.4f} ms/sample")
    return results, latency_results

def load_benchmark_results(path):
    data = np.load(path)
    snr_db = data["snr_db"]
    mean_wsr = data["mean_wsr"]
    return snr_db, mean_wsr
# =========================
# Main
# =========================
if __name__ == "__main__":
    np.random.seed(1234)
    torch.manual_seed(1234)

    # -------- load channels --------
    H_hist, H_fut = load_channel_data(hist_path, fut_path, hist_key=hist_key, fut_key=fut_key)
    channel_hist_seq_test, channel_tplusd_test = prepare_wmmse_channels(H_hist, H_fut,T,delta,device,DTYPE)

    Pxy_test, Vxy_test = load_posvel(Pxy_test_path, Vxy_test_path, pxy_key, vxy_key)
    mob_test = build_mobility_npgr_input(Pxy_test, Vxy_test, T=T)
    mob_mean, mob_std, _ = load_mobility_norm_stats(f"Weights/mobility_npgr_norm_stats_{scenario}.json")
    mob_test = normalize_mobility_test(mob_test, mob_mean, mob_std)
    mob_test = torch.tensor(mob_test, dtype=DTYPE, device=device)

    # -------- evaluate Predictive unfolded wmmse --------
    print("Predictive Unfolded WMMSE")
    wsr_predictive_model, latency_predictive  = evaluate_vs_snr_predictive_model(channel_tplusd_test,channel_hist_seq_test,mob_test,model_path,K,delta)
    print(f"Overall average predictive inference latency: {np.mean(latency_predictive):.4f} ms/sample")

    # -------- load benchmarks --------
    snr_llm4cp, wsr_llm4cp = load_benchmark_results(benchmark_LLM4CP)
    snr_kf, wsr_kf = load_benchmark_results(benchmark_KF)
    snr_genie, wsr_genie = load_benchmark_results(benchmark_genie)
    snr_current, wsr_current= load_benchmark_results(benchmark_cur_csi)

    # -------- plot --------
    plt.figure(figsize=(5.0, 3.93))
    plt.plot(snr_genie, wsr_genie, linestyle='--', marker='o',label='Genie-Aided CSI + WMMSE')
    plt.plot(snr_llm4cp, wsr_llm4cp, linestyle='--', marker='o',label='LLM4CP + WMMSE')
    plt.plot(SNR_range, wsr_predictive_model, linestyle='--', marker='o',label='Predictive Unfolded WMMSE')
    plt.plot(snr_kf, wsr_kf, linestyle='--', marker='o',label='KF + WMMSE')
    plt.plot(snr_current, wsr_current, linestyle='--', marker='o',label='Noisy Current CSI + WMMSE')
    plt.xlabel(r"Transmit SNR $P/\sigma^2$ [dB]", fontsize=9)
    plt.ylabel("WSR [bit/s/Hz]", fontsize=9)
    plt.grid(True)
    plt.legend()
    plt.xticks(SNR_range, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"results/WSR_vs_SNR_{scenario}.pdf",bbox_inches="tight")
    plt.show()

