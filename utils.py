import numpy as np
import torch
import hdf5storage
import json
from einops import rearrange
from Benchmarks.LLM4CP.GPT4CP import Model as LLMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# LLM config
llm_prev_len = 16 # history channels: H(n), H(n-1), ... , H(n-15)
llm_pred_len = 4  # future channels : H(n+1), ... , H(n+4)
num_coefficients = 32  # Pol * Mv * Mh
llm_patch_size = 4

def add_awgn(H, snr_db):
    noise_variance = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_variance / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
    channel_power = np.mean(np.abs(H) ** 2)
    noise = noise * np.sqrt(channel_power + 1e-12)
    return H + noise

def load_channel_data(hist_path, fut_path, hist_key="H_hist_c", fut_key="H_fut_c"):
    H_hist = hdf5storage.loadmat(hist_path)[hist_key]   # [S,N,hist_len,Pol,Mv,Mh] complex
    H_fut = hdf5storage.loadmat(fut_path)[fut_key]      # [S,N,fut_len,Pol,Mv,Mh] complex
    return H_hist, H_fut

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

def build_cache(model, H_past, user_weights, total_power):
    """
    H_past: [B,T-1,N,2M,2]
    Returns:
        cache["H"]          : [B,T-1,N,2M,2]
        cache["u"][l]       : [B,T-1,N,2,1]
        cache["w"][l]       : [B,T-1,N]
        cache["grad"][l][k] : [B,T-1,N,2M,1]
    """
    entries = []
    # Run Stage 1 independently at each historical time
    for tau in range(H_past.shape[1]):
        _, current_entry = model(H_t=H_past[:, tau],mob_features=None,cache=None,total_power=total_power,user_weights=user_weights,stage=1)
        entries.append(current_entry)

    # -------------------------
    # Stack along time dimension
    # -------------------------
    H_cache = torch.stack([entry["H"] for entry in entries],dim=1)
    u_cache = [torch.stack([entry["u"][l] for entry in entries],dim=1) for l in range(model.L)]
    w_cache = [torch.stack([entry["w"][l] for entry in entries],dim=1)for l in range(model.L)]
    grad_cache = [[torch.stack([entry["grad"][l][k] for entry in entries],dim=1) for k in range(model.K)] for l in range(model.L)]
    cache = {
        "H": H_cache,
        "u": u_cache,
        "w": w_cache,
        "grad": grad_cache
    }
    return cache

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

def normalize_mobility_train_val(train_mob, val_mob, eps=1e-6):
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

def prepare_wmmse_channels(H_hist, H_fut, T_seq, delta, device, dtype):
    """
    Inputs:
        H_hist : [S, N, hist_len, Pol, Mv, Mh] complex
        H_fut  : [S, N, fut_len,  Pol, Mv, Mh] complex
    Returns:
        H_hist_seq : [S, T_seq, N, 2M, 2]
        H_nplusd   : [S, N, 2M, 2]
    where M = Pol * Mv * Mh
    """
    # Complex channel -> real/imag user representation
    hist_user = channel_to_user_coefficients(H_hist)  # [S,N,hist_len,M,2]
    fut_user = channel_to_user_coefficients(H_fut)  # [S,N,fut_len,M,2]

    # Keep the last T_seq historical CSI samples
    hist_user_seq = hist_user[:, :, -T_seq:, :, :]  # [S,N,T_seq,M,2]

    # Convert historical sequence to real-valued WMMSE representation
    H_hist_seq = ri_sequence_to_wmmse_channel(hist_user_seq).to(device=device,dtype=dtype)  # [S,T_seq,N,2M,2]

    # Select H_{n+delta}
    H_future = fut_user[:, :, delta - 1, :, :]  # [S,N,M,2]

    # Convert future channel to real-valued WMMSE representation
    H_nplusd = ri_to_wmmse_channel(H_future).to(device=device,dtype=dtype)  # [S,N,2M,2]

    return H_hist_seq, H_nplusd

def channel_history_to_coefficient_sequences(H_hist, num_coefficients=32):
    """
    H_hist: [S, N, hist_len, Pol, Mv, Mh] complex

    Returns:
        x: [S*N*num_coefficients, hist_len, 2] float
        S: number of samples
        N: number of users
    """
    S, N, hist_len, Pol, Mv, Mh = H_hist.shape
    # [S,N,hist_len,Pol,Mv,Mh] -> [S*N,hist_len,C]
    H_hist = rearrange(H_hist,'s n t p mv mh -> (s n) t (p mv mh)')
    B, hist_len, num_channel_coeffs = H_hist.shape
    assert num_channel_coeffs == num_coefficients, f"Expected {num_coefficients} channel coefficients, "f"got {num_channel_coeffs}"
    # [B,hist_len,C] -> [B*C,hist_len,1]
    H_hist = rearrange(H_hist,'b t c -> (b c) t 1')
    # Complex -> real/imaginary representation
    x = np.zeros((B * num_coefficients, hist_len, 2),dtype=np.float32)
    x[:, :, 0] = H_hist[..., 0].real
    x[:, :, 1] = H_hist[..., 0].imag
    return torch.from_numpy(x), S, N

def coefficient_sequences_to_user_channels(pred_sequences,S,N,num_coefficients=32):
    """
    pred_sequences: [S*U*num_coefficients, fut_len, 2]
    Returns:
        pred_user_channels: [S, N, fut_len, num_coefficients, 2]
    """
    num_sequences = pred_sequences.shape[0]
    assert num_sequences == S * N * num_coefficients, ("Mismatch in total number of coefficient sequences")
    pred_user_channels = rearrange(pred_sequences,'(s n c) t ri -> s n t c ri',s=S,n=N,c=num_coefficients)
    return pred_user_channels

def channel_to_user_coefficients(H):
    """
    H: [S, N, seq_len, Pol, Mv, Mh] complex
    Returns:
        H_ri: [S, N, seq_len, num_coefficients, 2]
    """
    # [S,N,T,Pol,Mv,Mh] -> [S,N,T,C]
    H = rearrange(H,'s n t pol mv mh -> s n t (pol mv mh)')
    # Complex -> real/imaginary representation
    H_ri = np.stack([H.real, H.imag],axis=-1).astype(np.float32)
    return torch.from_numpy(H_ri)

def ri_to_wmmse_channel(h_ri):
    """
    Convert a complex channel represented by its real and imaginary
    parts to the real-valued matrix representation used by WMMSE.

    Input: h_ri: [B, N, M, 2]
    Returns: H: [B, N, 2M, 2]
    """
    h_real = h_ri[..., 0]   # [B,N,M]
    h_imag = h_ri[..., 1]   # [B,N,M]
    top = torch.stack([h_real, -h_imag],dim=-1)  # [B,N,M,2]
    bottom = torch.stack([h_imag, h_real],dim=-1)  # [B,N,M,2]
    H = torch.cat([top, bottom],dim=2)  # [B,N,2M,2]
    return H

def ri_sequence_to_wmmse_channel(h_seq_ri):
    """
    Input: h_seq_ri: [S, N, seq_len, M, 2]
    Returns: H_seq: [S, seq_len, N, 2M, 2]
    """
    h_real = h_seq_ri[..., 0]  # [S,N,seq_len,M]
    h_imag = h_seq_ri[..., 1]  # [S,N,seq_len,M]
    top = torch.stack([h_real, -h_imag],dim=-1)  # [S,N,seq_len,M,2]
    bottom = torch.stack([h_imag, h_real],dim=-1)  # [S,N,seq_len,M,2]
    H_seq = torch.cat([top, bottom],dim=3)  # [S,N,seq_len,2M,2]
    H_seq = rearrange(H_seq,'s n t m ri -> s t n m ri')  # [S,seq_len,N,2M,2]
    return H_seq

def ri_to_complex_channel(h_ri):
    """
    Convert a real-imaginary channel representation to complex form.
    Input: h_ri: [..., M, 2]
    Returns: h_complex: [..., M]
    """
    return h_ri[..., 0] + 1j * h_ri[..., 1]

def load_mobility_norm_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    C = len(stats["mean"])
    mean = np.array(stats["mean"], dtype=np.float32).reshape(1, C, 1, 1)
    std = np.array(stats["std"], dtype=np.float32).reshape(1, C, 1, 1)

    return mean, std, stats

def normalize_mobility_test(test_mob, mean, std, eps=1e-6):
    std = np.maximum(std, eps)
    return ((test_mob - mean) / std).astype(np.float32)

# =========================
# LLM inference
# =========================
def load_llm_model(weights_path):
    model = LLMModel(pred_len=llm_pred_len, prev_len=llm_prev_len, patch_size=llm_patch_size).to(device)
    state = torch.load(weights_path, map_location=device,weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def normalize_tensor(x, norm_stats):
    scale = norm_stats["scale"]
    return x / scale


def denormalize_tensor(x, norm_stats):
    scale = norm_stats["scale"]
    return x * scale

@torch.no_grad()
def predict_future_with_llm(model,H_hist_raw,norm_stats,num_coefficients=32,warmup_samples=20):
    """
    H_hist_raw:
        [S,N,hist_len,Pol,Mv,Mh] complex
    Returns:
        pred_userwise:
        [S,N,fut_len,32,2] in original (de-normalized) scale
    Also reports average online inference latency for one complete
    channel sample, i.e., all N users and all channel coefficients.
    """
    # ---------------------------------------------------------
    # Convert channel history to coefficient-wise sequences
    # ---------------------------------------------------------
    x_llm, S, N = channel_history_to_coefficient_sequences(H_hist_raw,num_coefficients=num_coefficients)# x_llm: [S*N*M, P, 2]

    # Normalize using TRAIN statistics
    x_llm_norm = normalize_tensor(x_llm, norm_stats)
    # Number of coefficient sequences belonging to one complete scene
    sequences_per_sample = N * num_coefficients
    pred_sequences = []
    total_inference_time_ms = 0.0
    timed_samples = 0
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    # ---------------------------------------------------------
    # Process one complete scene at a time
    # ---------------------------------------------------------
    for s in range(S):
        start_idx = s * sequences_per_sample
        end_idx = (s + 1) * sequences_per_sample
        # All N*M coefficient sequences of scene s
        x_sample = x_llm_norm[start_idx:end_idx].to(device)
        # [N*M, P, 2]
        measure_time = s >= warmup_samples
        if measure_time:
            torch.cuda.synchronize()
            starter.record()
        # -----------------------------------------------------
        # LLM4CP inference for the complete channel sample
        # -----------------------------------------------------
        y_sample = model(x_sample,None,None,None)# [N*M, fut_len, 2]
        if measure_time:
            ender.record()
            torch.cuda.synchronize()
            elapsed_ms = starter.elapsed_time(ender)
            total_inference_time_ms += elapsed_ms
            timed_samples += 1
        pred_sequences.append(y_sample.cpu())
    # ---------------------------------------------------------
    # Average online latency per complete channel sample
    # ---------------------------------------------------------
    avg_inference_time_ms = (total_inference_time_ms / timed_samples)
    print(f"LLM4CP average inference time: {avg_inference_time_ms:.4f} ms/sample")

    # ---------------------------------------------------------
    # Reconstruct prediction tensor
    # ---------------------------------------------------------
    pred_sequences = torch.cat(pred_sequences,dim=0)# [S*N*M, fut_len, 2]
    # De-normalize prediction
    pred_sequences = denormalize_tensor(pred_sequences,norm_stats)
    # [S*N*M,fut_len,2] -> [S,N,fut_len,M,2]
    pred_user_channels = coefficient_sequences_to_user_channels(pred_sequences,S,N,num_coefficients=num_coefficients)
    return pred_user_channels

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params

