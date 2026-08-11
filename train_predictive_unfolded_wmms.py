import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import hdf5storage
import json
from beamforming_utils import (make_user_weights,load_hist_fut,convert_to_user_channels,complex_seq_ri_to_wmmse_channel,complex_vector_ri_to_wmmse_channel)
from predictive_unfolded_wmmse import Predictive_UnfoldedWMMSE
from metrics import compute_WSR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
DTYPE = torch.float32

def load_posvel(Pxy_path, Vxy_path, pxy_key="Pxy", vxy_key="Vxy"):
    """
    Load position and velocity history from .mat files.
    Expected:
        Pxy : [S, T, N, 2]
        Vxy : [S, T, N, 2]
    """
    Pxy = hdf5storage.loadmat(Pxy_path)[pxy_key]
    Vxy = hdf5storage.loadmat(Vxy_path)[vxy_key]
    return Pxy, Vxy

def build_mobility_npgr_input(Pxy, Vxy, T=5):
    if T > Pxy.shape[1]:
        raise ValueError(f"T={T} exceeds available history length {Pxy.shape[1]}.")

    P = Pxy[:, -T:, :, :]   # [S,T,N,2]
    V = Vxy[:, -T:, :, :]   # [S,T,N,2]

    # Differences aligned to times tau = n-T+2, ..., n
    # dP_tau = P_tau - P_{tau-1}
    dP = P[:, 1:, :, :] - P[:, :-1, :, :]      # [S,T-1,N,2]
    # dV_tau = V_tau - V_{tau-1}
    dV = V[:, 1:, :, :] - V[:, :-1, :, :]      # [S,T-1,N,2]
    # Velocity at the same time tau
    V_aligned = V[:, 1:, :, :]                 # [S,T-1,N,2]
    # concatenate features per time:
    # [dP_x,dP_y,V_x,V_y,dV_x,dV_y]
    mob = np.concatenate([dP, V_aligned, dV], axis=-1)  # [S,T-1,N,6]
    mob = np.transpose(mob, (0, 3, 2, 1))              # [S,6,N,T-1]
    return mob.astype(np.float32)

def normalize_mobility(train_mob, val_mob, eps=1e-6):
    """
    Inputs:
        train_mob : [S_train, D, N, T-1]
        val_mob   : [S_val,   D, N, T-1]
    Per-channel normalization using train statistics only.
    Statistics are computed over samples, users and time.
    """
    mean = train_mob.mean(axis=(0, 2, 3), keepdims=True)   # [1,C,1,1]
    std = train_mob.std(axis=(0, 2, 3), keepdims=True)     # [1,C,1,1]
    std = np.maximum(std, eps)
    train_norm = (train_mob - mean) / std
    val_norm = (val_mob - mean) / std
    norm_stats = {
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
        "input_shape": list(train_mob.shape[1:]),  # [C, N, T-1]
        "channels": ["dP_x","dP_y","V_x","V_y","dV_x","dV_y"]}

    return train_norm.astype(np.float32),val_norm.astype(np.float32),norm_stats

def save_norm_stats(norm_stats, save_path="Weights/mobility_npgr_norm_stats.json"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=4)
    print(f"Saved mobility NPGR normalization stats to: {save_path}")

def train_pgd_step_sizes(channel_nplusd_train, channel_hist_seq_train,
     channel_nplusd_val, channel_hist_seq_val, user_weights, num_epochs, batch_size, K, L, noise_power, SNR_range, delta, weights_dir="Weights"):
    print(f"\n************ Starting stage 1 of training ************")
    S_train = channel_hist_seq_train.shape[0]
    S_val = channel_hist_seq_val.shape[0]
    _,_, N, twoM, _ = channel_hist_seq_train.shape
    gamma_prev = None
    mob_features = None
    results = {}

    for l in range(1, L + 1):
        # ------------------------------------------------------------
        # Progressive training: freeze previous layers, train only new layer
        # gamma_prev: [l-1, K] (no grad)
        # gamma_new : [1, K] (trainable)
        # ------------------------------------------------------------
        if l == 1:
            Gamma = torch.ones((1, K), dtype=DTYPE, device=device)
        else:
            Gamma = torch.zeros((l, K), dtype=DTYPE, device=device)
            Gamma[:l - 1] = gamma_prev.to(device=device, dtype=DTYPE)

        model = Predictive_UnfoldedWMMSE(l, N, K, noise_power, Gamma, delta).to(device)
        model.NPGRs.requires_grad_(False)
        model.Gamma.requires_grad_(True)
        optimizer = optim.Adam([model.Gamma], lr=1e-4)
        best_val_loss = float("inf")
        best_state = None
        train_loss_history = []
        val_loss_history = []
        epochs_no_improvement = 0
        patience = 30

        print(f"\n************ Starting PGD step sizes training with L={l} ************")
        for epoch in range(num_epochs):
            model.train()
            perm = torch.randperm(S_train,device=channel_hist_seq_train.device)
            epoch_loss_sum = 0.0
            epoch_samples = 0
            for start in range(0, S_train, batch_size):
                end = min(start + batch_size, S_train)
                idx = perm[start:end]

                channel_hist_seq_batch = channel_hist_seq_train[idx]  # [B,N,2M,2]
                ch_nplusd_batch = channel_nplusd_train[idx]  # [B,N,2M,2]
                snr_db = np.random.uniform(min(SNR_range), max(SNR_range))
                total_power_batch = 10 ** (snr_db / 10)

                V_final = model(mob_features, channel_hist_seq_batch, user_weights, total_power_batch, stage=1)
                loss = - compute_WSR(noise_power, user_weights, ch_nplusd_batch, V_final)

                optimizer.zero_grad()
                loss.backward()
                #update only gamma in new layer
                if l > 1 and model.Gamma.grad is not None:
                    with torch.no_grad():
                        model.Gamma.grad[:l - 1].zero_()

                optimizer.step()
                B = ch_nplusd_batch.shape[0]
                epoch_loss_sum += loss.item() * B
                epoch_samples += B

            mean_train_loss = epoch_loss_sum / epoch_samples
            train_loss_history.append(mean_train_loss)
            # ---------------- VALIDATION ----------------
            model.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                val_count = 0

                for start in range(0, S_val, batch_size):
                    end = min(start + batch_size, S_val)

                    channel_hist_seq_batch = channel_hist_seq_val[start:end]
                    ch_nplusd_batch = channel_nplusd_val[start:end]
                    B = ch_nplusd_batch.shape[0]

                    for snr_db in SNR_range:
                        total_power_batch = 10 ** (snr_db / 10)
                        V_final = model(mob_features, channel_hist_seq_batch, user_weights, total_power_batch, stage=1)
                        val_loss = -compute_WSR(noise_power,user_weights,ch_nplusd_batch,V_final)
                        val_loss_sum += val_loss.item() * B
                        val_count += B

            mean_val_loss = val_loss_sum / val_count
            val_loss_history.append(mean_val_loss)

            if mean_val_loss < best_val_loss - 1e-4:
                best_val_loss = mean_val_loss
                epochs_no_improvement = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"EPOCH {epoch + 1}/{num_epochs} | TRAIN LOSS ={mean_train_loss:.4f} | VAL LOSS={mean_val_loss:.4f}")

                print("Step sizes:")
                for layer in range(l):
                    print(f"layer{layer + 1}:")
                    for k in range(K):
                        val = model.Gamma[layer, k].detach().cpu().item()
                        print(f"  step{k + 1}: {val:.5f}")

        #keep the best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        gamma_prev = model.Gamma.detach().clone()
        results[l] = {"Gamma": gamma_prev.cpu(),"best_val_loss": best_val_loss}
        print(f"Best validation loss for L={l}: {best_val_loss:.4f}")
        plt.figure()
        plt.plot(train_loss_history, label="Train Loss")
        plt.plot(val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Stage 1 training with L={l}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    os.makedirs(weights_dir, exist_ok=True)
    final_gamma_path = os.path.join(weights_dir,f"PGD_step_sizes_L{L}_K{K}_stage1.pt")
    torch.save(results[L]["Gamma"], final_gamma_path)
    print(f"Saved final PGD step sizes to: {final_gamma_path}")

def train_NPGRs_and_tune_PGD_step_sizes(channel_nplusd_train, channel_hist_seq_train,mob_features_train,
     channel_nplusd_val, channel_hist_seq_val, mob_features_val, user_weights, num_epochs, batch_size, K, L, noise_power, SNR_range, delta, weights_dir="Weights"):
    print(f"\n************ Starting stage 2 of training ************")

    S_train = channel_nplusd_train.shape[0]
    S_val = channel_nplusd_val.shape[0]
    _, N, twoM, _ = channel_nplusd_train.shape
    gamma_path = os.path.join(weights_dir,f"PGD_step_sizes_L{L}_K{K}_stage1.pt")
    Gamma = torch.load(gamma_path,map_location=device,weights_only=False).to(device=device, dtype=DTYPE)
    model = Predictive_UnfoldedWMMSE(L, N,K,noise_power, Gamma, delta).to(device)

    model.NPGRs.requires_grad_(True)
    model.Gamma.requires_grad_(True)

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    best_val_loss = float("inf")
    best_state = None
    train_loss_history = []
    val_loss_history = []
    epochs_no_improvement = 0
    patience = 100
    print(f"\n************ Starting Predictive Unfolded WMMSE training with L={L} layers ************")

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(S_train,device=channel_hist_seq_train.device)
        epoch_loss_sum = 0.0
        epoch_samples = 0
        for start in range(0, S_train, batch_size):
            end = min(start + batch_size, S_train)
            idx = perm[start:end]

            ch_nplusd_batch = channel_nplusd_train[idx]  # [B,N,2M,2]
            channel_hist_seq_batch = channel_hist_seq_train[idx]
            mob_features_train_batch = mob_features_train[idx]

            snr_db = np.random.uniform(min(SNR_range), max(SNR_range))
            total_power_batch = 10 ** (snr_db / 10)
            V_final = model(mob_features_train_batch, channel_hist_seq_batch, user_weights, total_power_batch, stage=2)
            loss = - compute_WSR(noise_power, user_weights, ch_nplusd_batch, V_final)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            B = ch_nplusd_batch.shape[0]
            epoch_loss_sum += loss.item() * B
            epoch_samples += B

        mean_train_loss = epoch_loss_sum / epoch_samples
        train_loss_history.append(mean_train_loss)
        # ---------------- VALIDATION ----------------
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_count = 0
            for start in range(0, S_val, batch_size):
                end = min(start + batch_size, S_val)

                ch_nplusd_batch = channel_nplusd_val[start:end]
                channel_hist_seq_batch = channel_hist_seq_val[start:end]
                mob_features_val_batch = mob_features_val[start:end]
                B = ch_nplusd_batch.shape[0]

                for snr_db in SNR_range:
                    total_power_batch = 10 ** (snr_db / 10)
                    V_final = model(mob_features_val_batch, channel_hist_seq_batch, user_weights, total_power_batch, stage=2)
                    val_loss = -compute_WSR(noise_power, user_weights, ch_nplusd_batch, V_final)
                    val_loss_sum += val_loss.item() * B
                    val_count += B

        mean_val_loss = val_loss_sum / val_count
        val_loss_history.append(mean_val_loss)

        if mean_val_loss < best_val_loss - 1e-4:
            best_val_loss = mean_val_loss
            epochs_no_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"EPOCH {epoch + 1}/{num_epochs} | TRAIN LOSS ={mean_train_loss:.4f} | VAL LOSS={mean_val_loss:.4f}")

            print("Step sizes:")
            for l in range(L):
                print(f"layer{l + 1}:")
                for k in range(K):
                    val = model.Gamma[l, k].detach().cpu().item()
                    print(f"  step{k + 1}: {val:.5f}")

    print(f"Best validation Loss for L={L}: {best_val_loss:.4f}")
    plt.figure()
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Stage 2 training with L={L}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if best_state is not None:
        model.load_state_dict(best_state)

    # save checkpoint for this depth
    os.makedirs(weights_dir, exist_ok=True)
    layer_save_path = os.path.join(weights_dir, f"Predictive_unfolded_WMMSE_model_L{L}.pt")
    torch.save({"L": L,"model_state": model.state_dict(),"Gamma": model.Gamma.detach().cpu(),"best_val_loss": best_val_loss}, layer_save_path)
    print(f"Saved best checkpoint for L={L} to: {layer_save_path}")

def train_predictive_unfolded_wmmse(channel_nplusd_train, channel_hist_seq_train, mob_features_train, channel_nplusd_val, channel_hist_seq_val, mob_features_val, user_weights, num_epochs, batch_size, K, L, noise_power, SNR_range, delta, weights_dir="Weights"):
    #Stage 1
    train_pgd_step_sizes(channel_nplusd_train, channel_hist_seq_train,
                         channel_nplusd_val, channel_hist_seq_val, user_weights, num_epochs, batch_size, K, L, noise_power, SNR_range,delta, weights_dir=weights_dir)
    #Stage 2
    train_NPGRs_and_tune_PGD_step_sizes(channel_nplusd_train, channel_hist_seq_train, mob_features_train,
                                        channel_nplusd_val, channel_hist_seq_val, mob_features_val, user_weights, num_epochs, batch_size, K, L, noise_power, SNR_range,delta, weights_dir=weights_dir)


if __name__ == "__main__":
    np.random.seed(1234)
    torch.manual_seed(1234)

    T_seq = 5
    delta = 1
    K = 4
    L = 5
    batch_size = 64
    num_epochs = 1000
    SNR_range = [0, 2.5, 5, 7.5, 10, 12.5]
    noise_power = torch.tensor(1.0,dtype=DTYPE,device=device)

    # -------- Paths --------
    hist_path_train = "data/UMa/train_val/H_hist_train_noisy.mat"
    fut_path_train = "data/UMa/train_val/H_fut_train.mat"
    hist_path_val = "data/UMa/train_val/H_hist_val_noisy.mat"
    fut_path_val = "data/UMa/train_val/H_fut_val.mat"

    Pxy_train_path = "data/UMa/train_val/Pxy_hist_train.mat"
    Vxy_train_path = "data/UMa/train_val/Vxy_hist_train.mat"
    Pxy_val_path = "data/UMa/train_val/Pxy_hist_val.mat"
    Vxy_val_path = "data/UMa/train_val/Vxy_hist_val.mat"

    # -------- Load channels --------
    H_hist_train, H_fut_train = load_hist_fut(hist_path_train,fut_path_train,hist_key="H_hist_train",fut_key="H_fut_train")
    H_hist_val, H_fut_val = load_hist_fut(hist_path_val,fut_path_val,hist_key="H_hist_val",fut_key="H_fut_val")
    hist_train = convert_to_user_channels(H_hist_train)
    hist_val = convert_to_user_channels(H_hist_val)
    future_train = convert_to_user_channels(H_fut_train)
    future_val = convert_to_user_channels(H_fut_val)

    # -------- Historical channel sequence --------
    hist_train_seq = hist_train[:, :, -T_seq:, :, :]
    hist_val_seq = hist_val[:, :, -T_seq:, :, :]
    channel_hist_seq_train = (complex_seq_ri_to_wmmse_channel(hist_train_seq).to(device=device, dtype=DTYPE))
    channel_hist_seq_val = (complex_seq_ri_to_wmmse_channel(hist_val_seq).to(device=device, dtype=DTYPE))

    # -------- Future channel H_{n+delta} --------
    channel_nplusd_train = (complex_vector_ri_to_wmmse_channel(future_train[:, :, delta - 1, :, :]).to(device=device, dtype=DTYPE))
    channel_nplusd_val = (complex_vector_ri_to_wmmse_channel(future_val[:, :, delta - 1, :, :]).to(device=device, dtype=DTYPE))

    # -------- Mobility --------
    Pxy_train, Vxy_train = load_posvel(Pxy_train_path,Vxy_train_path,pxy_key="Pxy_train",vxy_key="Vxy_train")
    Pxy_val, Vxy_val = load_posvel(Pxy_val_path,Vxy_val_path,pxy_key="Pxy_val",vxy_key="Vxy_val")
    mob_train = build_mobility_npgr_input(Pxy_train,Vxy_train,T=T_seq)  # [S_train,6,N,T_seq-1]
    mob_val = build_mobility_npgr_input(Pxy_val,Vxy_val,T=T_seq)  # [S_val,6,N,T_seq-1]
    mob_train, mob_val, mob_stats = normalize_mobility(mob_train,mob_val)
    save_norm_stats(mob_stats)
    mob_train = torch.from_numpy(mob_train).to(device=device,dtype=DTYPE)
    mob_val = torch.from_numpy(mob_val).to(device=device,dtype=DTYPE)

    # -------- User weights --------
    N = channel_hist_seq_train.shape[2]
    user_weights = torch.ones((1, N, 1),device=device,dtype=DTYPE)

    # -------- Train --------
    train_predictive_unfolded_wmmse(channel_nplusd_train,channel_hist_seq_train,mob_train,channel_nplusd_val,channel_hist_seq_val,mob_val,user_weights,
        num_epochs=num_epochs,batch_size=batch_size,K=K,L=L,noise_power=noise_power,SNR_range=SNR_range,delta=delta)