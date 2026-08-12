import numpy as np
import torch
import hdf5storage
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset
from Benchmarks_code.LLM4CP.GPT4CP import Model as LLMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# LLM config
llm_prev_len = 16 #history channels: H(n), H(n-1), ... , H(n-15)
llm_pred_len = 4  #future channels : H(n+1), ... , H(n+4)
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

    Input:
        h_ri: [B, N, M, 2]

    Returns:
        H: [B, N, 2M, 2]
    """
    h_real = h_ri[..., 0]   # [B,N,M]
    h_imag = h_ri[..., 1]   # [B,N,M]
    top = torch.stack([h_real, -h_imag],dim=-1)  # [B,N,M,2]
    bottom = torch.stack([h_imag, h_real],dim=-1)  # [B,N,M,2]
    H = torch.cat([top, bottom],dim=2)  # [B,N,2M,2]
    return H

def ri_sequence_to_wmmse_channel(h_seq_ri):
    """
    Input:
        h_seq_ri: [S, N, seq_len, M, 2]
    Returns:
        H_seq: [S, seq_len, N, 2M, 2]
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
    Input:
        h_ri: [..., M, 2]
    Returns:
        h_complex: [..., M]
    """
    return h_ri[..., 0] + 1j * h_ri[..., 1]

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
    mean = norm_stats["mean"].to(x.device).view(*([1] * (x.ndim - 1)), -1)
    std = norm_stats["std"].to(x.device).view(*([1] * (x.ndim - 1)), -1)
    return (x - mean) / std

def denormalize_tensor(x, norm_stats):
    """
    x: [..., C]
    """
    mean = norm_stats["mean"].to(x.device).view(*([1] * (x.ndim - 1)), -1)
    std = norm_stats["std"].to(x.device).view(*([1] * (x.ndim - 1)), -1)
    return x * std + mean

@torch.no_grad()
def predict_future_with_llm(model, H_hist_raw, norm_stats, batch_size=512, num_coefficients=32):
    """
    H_hist_raw: [S,N,hist_len,Pol,Mv,Mh] complex
    Returns:
        pred_userwise: [S,N,fut_len,32,2]   in original (de-normalized) scale
    """
    x_llm, S, N = channel_history_to_coefficient_sequences(H_hist_raw, num_coefficients=num_coefficients)   # [S*N*32,P,2]
    # normalize input using TRAIN stats
    x_llm_norm = normalize_tensor(x_llm, norm_stats)
    loader = DataLoader(TensorDataset(x_llm_norm), batch_size=batch_size, shuffle=False)

    pred_sequences = []
    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        y_batch = model(x_batch, None, None, None)   # [B,fut_len,2] normalized
        pred_sequences.append(y_batch.cpu())

    pred_sequences = torch.cat(pred_sequences, dim=0)                  # [S*N*32,fut_len,2]

    # de-normalize prediction back to original scale
    pred_sequences = denormalize_tensor(pred_sequences, norm_stats)    # [S*N*32,fut_len,2]
    pred_user_channels  = coefficient_sequences_to_user_channels(pred_sequences, S, N, num_coefficients=num_coefficients)  # [S,N,fut_len,32,2]

    return pred_user_channels

