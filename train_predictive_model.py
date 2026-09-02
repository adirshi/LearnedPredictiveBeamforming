import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import json
from utils import load_channel_data, prepare_wmmse_channels, load_posvel,build_mobility_npgr_input,normalize_mobility_train_val,build_cache
from predictive_unfolded_wmmse import Predictive_UnfoldedWMMSE
from metrics import compute_WSR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
DTYPE = torch.float32

def save_norm_stats(norm_stats, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=4)
    print(f"Saved mobility NPGR normalization stats to: {save_path}")

def train_stage1(H_future_train, H_t_train,
     H_future_val, H_t_val, user_weights, num_epochs, batch_size, K, L, noise_power, tx_snr_range, scenario, delta, weights_dir="Weights"):
    print("\n************ Starting stage 1 ************")
    S_train = H_t_train.shape[0]
    S_val = H_t_val.shape[0]
    N = H_t_train.shape[1]
    gamma_prev = None

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
            Gamma[:l - 1] = gamma_prev

        model = Predictive_UnfoldedWMMSE(l, N, K, noise_power, Gamma, delta).to(device)
        model.NPGRs.requires_grad_(False)
        model.Gamma.requires_grad_(True)
        optimizer = optim.Adam([model.Gamma], lr=1e-4)
        best_val_loss = float("inf")
        best_gamma = None
        epochs_no_improvement = 0
        patience = 20

        for epoch in range(num_epochs):
            model.train()
            perm = torch.randperm(S_train,device=H_t_train.device)
            epoch_loss_sum = 0.0
            epoch_samples = 0
            for start in range(0, S_train, batch_size):
                end = min(start + batch_size, S_train)
                idx = perm[start:end]

                H_t_batch = H_t_train[idx]
                H_future_batch = H_future_train[idx]

                tx_snr_db = np.random.uniform(min(tx_snr_range), max(tx_snr_range))
                total_power_batch = 10 ** (tx_snr_db / 10)
                V_final, _ = model(H_t=H_t_batch,mob_features=None,cache=None,total_power=total_power_batch,user_weights=user_weights,stage=1)
                loss = - compute_WSR(noise_power, user_weights, H_future_batch, V_final)

                optimizer.zero_grad()
                loss.backward()
                #update only gamma in new layer
                if l > 1 and model.Gamma.grad is not None:
                    with torch.no_grad():
                        model.Gamma.grad[:l - 1].zero_()

                optimizer.step()
                B = H_future_batch.shape[0]
                epoch_loss_sum += loss.item() * B
                epoch_samples += B

            mean_train_loss = epoch_loss_sum / epoch_samples
            # ---------------- VALIDATION ----------------
            model.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                val_count = 0

                for start in range(0, S_val, batch_size):
                    end = min(start + batch_size, S_val)

                    H_t_batch = H_t_val[start:end]
                    H_future_batch = H_future_val[start:end]
                    B = H_future_batch.shape[0]

                    for tx_snr_db in tx_snr_range:
                        total_power_batch = 10 ** (tx_snr_db / 10)
                        V_final, _ = model(H_t=H_t_batch,mob_features=None,cache=None,total_power=total_power_batch,user_weights=user_weights,stage=1)
                        val_loss = - compute_WSR(noise_power,user_weights,H_future_batch,V_final)
                        val_loss_sum += val_loss.item() * B
                        val_count += B

            mean_val_loss = val_loss_sum / val_count
            if mean_val_loss < best_val_loss - 1e-4:
                best_val_loss = mean_val_loss
                epochs_no_improvement = 0
                best_gamma = model.Gamma.detach().clone()
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"EPOCH {epoch + 1}/{num_epochs} | TRAIN LOSS ={mean_train_loss:.4f} | VAL LOSS={mean_val_loss:.4f}")

        # keep the best step sizes
        if best_gamma is not None:
            with torch.no_grad():
                model.Gamma.copy_(best_gamma)

        gamma_prev = model.Gamma.detach().clone()
        print(f"Best validation loss for L={l}: {best_val_loss:.4f}")

    os.makedirs(weights_dir, exist_ok=True)
    final_gamma_path = os.path.join(weights_dir,f"PGD_step_sizes_{scenario}_delta{delta}_L{L}_stage1.pt")
    torch.save(gamma_prev.cpu(),final_gamma_path)
    print(f"Saved final PGD step sizes to: {final_gamma_path}")

def train_stage2(H_future_train, H_hist_train_seq, mob_features_train,
     H_future_val, H_hist_seq_val, mob_features_val, user_weights, num_epochs, batch_size, K, L, noise_power, tx_snr_range, scenario, delta, weights_dir="Weights"):
    print("\n************ Starting stage 2 ************")

    S_train = H_hist_train_seq.shape[0]
    S_val = H_hist_seq_val.shape[0]
    N = H_hist_train_seq.shape[2]
    gamma_path = os.path.join(weights_dir,f"PGD_step_sizes_{scenario}_delta{delta}_L{L}_stage1.pt")
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
    patience = 40

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(S_train,device=H_hist_train_seq.device)
        epoch_loss_sum = 0.0
        epoch_samples = 0
        for start in range(0, S_train, batch_size):
            end = min(start + batch_size, S_train)
            idx = perm[start:end]

            H_future_batch = H_future_train[idx]
            H_hist_seq_batch = H_hist_train_seq[idx]
            mob_features_train_batch = mob_features_train[idx]

            H_past = H_hist_seq_batch[:, :-1] # t-T+1, ..., t-1
            H_t = H_hist_seq_batch[:, -1]     # t

            tx_snr_db = np.random.uniform(min(tx_snr_range), max(tx_snr_range))
            total_power_batch = 10 ** (tx_snr_db / 10)
            cache = build_cache(model=model,H_past=H_past,user_weights=user_weights,total_power=total_power_batch)
            V_final, _ = model(H_t,mob_features_train_batch,cache,total_power_batch,user_weights,stage=2)
            loss = - compute_WSR(noise_power, user_weights, H_future_batch, V_final)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            B = H_future_batch.shape[0]
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

                H_future_batch = H_future_val[start:end]
                H_hist_seq_batch = H_hist_seq_val[start:end]
                mob_features_val_batch = mob_features_val[start:end]
                H_past = H_hist_seq_batch[:, :-1]  # t-T+1, ..., t-1
                H_t = H_hist_seq_batch[:, -1]  # t
                B = H_future_batch.shape[0]

                for tx_snr_db in tx_snr_range:
                    total_power_batch = 10 ** (tx_snr_db / 10)
                    cache = build_cache(model=model, H_past=H_past, user_weights=user_weights,total_power=total_power_batch)
                    V_final, _ = model(H_t, mob_features_val_batch, cache, total_power_batch,user_weights,stage=2)
                    val_loss = -compute_WSR(noise_power, user_weights, H_future_batch, V_final)
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
    layer_save_path = os.path.join(weights_dir, f"Predictive_unfolded_WMMSE_model_{scenario}_delta{delta}_L{L}.pt")
    torch.save({"L": L,"model_state": model.state_dict(),"Gamma": model.Gamma.detach().cpu(),"best_val_loss": best_val_loss}, layer_save_path)
    print(f"Saved best checkpoint for L={L} to: {layer_save_path}")

def train_predictive_unfolded_wmmse(H_future_train, H_hist_train_seq, mob_features_train, H_future_val, H_hist_seq_val, mob_features_val, user_weights, num_epochs, batch_size, K, L, noise_power, tx_snr_range, scenario, delta, weights_dir="Weights"):
    #Stage 1
    train_stage1(H_future_train, H_hist_train_seq[:,-1],
                         H_future_val, H_hist_seq_val[:,-1], user_weights, num_epochs, batch_size, K, L, noise_power,tx_snr_range, scenario, delta, weights_dir=weights_dir)
    #Stage 2
    train_stage2(H_future_train, H_hist_train_seq, mob_features_train,
                                        H_future_val, H_hist_seq_val, mob_features_val, user_weights, num_epochs, batch_size, K, L, noise_power,tx_snr_range,scenario,delta, weights_dir=weights_dir)

if __name__ == "__main__":
    np.random.seed(1234)
    torch.manual_seed(1234)
    scenario = "RMa"  # "UMa" or "RMa"
    T = 5
    delta = 1
    N = 4
    K = 4
    L = 5
    batch_size = 64
    num_epochs = 2000
    tx_snr_range = [0, 2.5, 5, 7.5, 10, 12.5]
    noise_power = torch.tensor(1.0,dtype=DTYPE,device=device)

    # -------- Channel Paths --------
    hist_path_train = f"data/{scenario}/train_val/H_hist_train_noisy.mat"
    fut_path_train = f"data/{scenario}/train_val/H_fut_train.mat"
    hist_path_val = f"data/{scenario}/train_val/H_hist_val_noisy.mat"
    fut_path_val = f"data/{scenario}/train_val/H_fut_val.mat"

    # -------- Mobility features Paths --------
    Pxy_train_path = f"data/{scenario}/train_val/Pxy_hist_train.mat"
    Vxy_train_path = f"data/{scenario}/train_val/Vxy_hist_train.mat"
    Pxy_val_path = f"data/{scenario}/train_val/Pxy_hist_val.mat"
    Vxy_val_path = f"data/{scenario}/train_val/Vxy_hist_val.mat"

    # -------- Load channels and convert from complex to real representation --------
    H_hist_train, H_fut_train = load_channel_data(hist_path_train,fut_path_train,hist_key="H_hist_train",fut_key="H_fut_train") # [S_train,N,hist_len,Pol,Mv,Mh] complex , [S_train,N,fut_len,Pol,Mv,Mh] complex
    H_hist_val, H_fut_val = load_channel_data(hist_path_val,fut_path_val,hist_key="H_hist_val",fut_key="H_fut_val") # [S_val,N,hist_len,Pol,Mv,Mh] complex , [S_val,N,fut_len,Pol,Mv,Mh] complex

    H_hist_seq_train, H_future_train = prepare_wmmse_channels(H_hist_train,H_fut_train,T,delta,device,DTYPE) # [S_train,T,N,2M,2], [S_train,N,2M,2] real
    H_hist_seq_val, H_future_val = prepare_wmmse_channels(H_hist_val,H_fut_val,T,delta,device,DTYPE) # [S_val,T,N,2M,2], [S_val,N,2M,2] real

    # -------- Mobility --------
    Pxy_train, Vxy_train = load_posvel(Pxy_train_path,Vxy_train_path,pxy_key="Pxy_train",vxy_key="Vxy_train")
    Pxy_val, Vxy_val = load_posvel(Pxy_val_path,Vxy_val_path,pxy_key="Pxy_val",vxy_key="Vxy_val")
    mob_train = build_mobility_npgr_input(Pxy_train,Vxy_train,T)  # [S_train,6,N,T-1]
    mob_val = build_mobility_npgr_input(Pxy_val,Vxy_val,T)  # [S_val,6,N,T-1]
    mob_train, mob_val, mob_stats = normalize_mobility_train_val(mob_train,mob_val)
    save_norm_stats(mob_stats,save_path=f"Weights/mobility_npgr_norm_stats_{scenario}.json")
    mob_train = torch.from_numpy(mob_train).to(device=device,dtype=DTYPE)
    mob_val = torch.from_numpy(mob_val).to(device=device,dtype=DTYPE)

    user_weights = torch.ones((1, N, 1),device=device,dtype=DTYPE)

    train_predictive_unfolded_wmmse(H_future_train,H_hist_seq_train,mob_train,H_future_val,H_hist_seq_val,mob_val,user_weights,
        num_epochs,batch_size,K,L,noise_power,tx_snr_range,scenario,delta)

