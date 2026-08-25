import numpy as np
import hdf5storage
from utils import add_awgn

def make_noisy_hist_file(input_path, key, output_path, snr_range=(5.0, 20.0), seed=42):
    np.random.seed(seed)
    data = hdf5storage.loadmat(input_path)
    H_hist = data[key]

    # Expected: [Scenes, Users, hist_len, Pol, Mv, Mh]
    S, N = H_hist.shape[0], H_hist.shape[1]
    H_noisy = H_hist.copy()
    measurement_noise_var = np.zeros((S, N), dtype=np.float32)
    for s in range(S):
        for n in range(N):
            csi_snr_db  = np.random.uniform(snr_range[0], snr_range[1])
            channel_power = np.mean(np.abs(H_hist[s, n, ...]) ** 2)
            # Exact measurement-noise variance
            R = channel_power * 10 ** (-csi_snr_db / 10)
            measurement_noise_var[s, n] = R
            H_noisy[s, n, ...] = add_awgn(H_hist[s, n, ...], csi_snr_db)

    hdf5storage.savemat(output_path,{key: H_noisy,"measurement_noise_var": measurement_noise_var,},format="7.3")
    print(f"Saved: {output_path}")
    print(f"Shape: {H_noisy.shape}")

if __name__ == "__main__":
    scenario = "RMa"  # "UMa" or "RMa"
    files = [
        (f"data/{scenario}/train_val/H_hist_train.mat",
        "H_hist_train",
        f"data/{scenario}/train_val/H_hist_train_noisy.mat",
        42),

        (f"data/{scenario}/train_val/H_hist_val.mat",
        "H_hist_val",
        f"data/{scenario}/train_val/H_hist_val_noisy.mat",
        43),

        (f"data/{scenario}/test/H_hist_test.mat",
        "H_hist_test",
        f"data/{scenario}/test/H_hist_test_noisy.mat",
        44)]

    for input_path, key, output_path, seed in files:
        make_noisy_hist_file(input_path=input_path, key=key,output_path=output_path,snr_range=(5.0, 20.0),seed=seed)
