import torch.utils.data as data
import torch
import numpy as np
import hdf5storage
from einops import rearrange
from numpy import random


def noise(H, snr_db):
    sigma = 10 ** (-snr_db / 10)
    noise_term = np.sqrt(sigma / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
    noise_term = noise_term * np.sqrt(np.mean(np.abs(H) ** 2) + 1e-12)
    return H + noise_term

def LoadBatch_narrowband(H, num=32):
    """
    H:   [B, T, mul] complex
    out: [B*num, T, 2*(mul/num)] real
    """
    B, T, mul = H.shape
    H = rearrange(H, 'b t (k a) -> (b a) t k', a=num)

    H_real = np.zeros([B * num, T, mul // num, 2], dtype=np.float32)
    H_real[:, :, :, 0] = H.real
    H_real[:, :, :, 1] = H.imag
    H_real = H_real.reshape([B * num, T, (mul // num) * 2])

    return torch.tensor(H_real, dtype=torch.float32)


def Transform_back_to_complex(H, last_dim_is_two=True):
    """
    Convert real-valued representation back to complex.
    """
    if last_dim_is_two:
        return torch.complex(H[..., 0], H[..., 1])
    raise NotImplementedError("Only last_dim_is_two=True is implemented.")


class Dataset_Pro(data.Dataset):
    def __init__(self, prev, fut):
        super(Dataset_Pro, self).__init__()
        self.prev = prev
        self.fut = fut

    def __getitem__(self, index):
        return self.fut[index].float(), self.prev[index].float()

    def __len__(self):
        return self.fut.shape[0]


def compute_norm_stats(prev_tensor, fut_tensor, eps=1e-6):
    """
    prev_tensor: [N, P, C]
    fut_tensor : [N, L, C]

    We compute normalization from TRAIN only, jointly on prev+fut.
    """
    joint = torch.cat([prev_tensor.reshape(-1, prev_tensor.shape[-1]),
                       fut_tensor.reshape(-1, fut_tensor.shape[-1])], dim=0)

    mean = joint.mean(dim=0, keepdim=True)   # [1, C]
    std = joint.std(dim=0, keepdim=True)     # [1, C]
    std = torch.clamp(std, min=eps)

    norm_stats = {
        "mean": mean.squeeze(0).cpu(),
        "std": std.squeeze(0).cpu(),
    }
    return norm_stats


def apply_normalization(prev_tensor, fut_tensor, norm_stats):
    """
    prev_tensor: [N, P, C]
    fut_tensor : [N, L, C]
    """
    mean = norm_stats["mean"].view(1, 1, -1)
    std = norm_stats["std"].view(1, 1, -1)

    prev_norm = (prev_tensor - mean) / std
    fut_norm = (fut_tensor - mean) / std

    return prev_norm, fut_norm


def denormalize_tensor(x, norm_stats):
    """
    x: [B, T, C]
    """
    mean = norm_stats["mean"].to(x.device).view(1, 1, -1)
    std = norm_stats["std"].to(x.device).view(1, 1, -1)
    return x * std + mean


def build_dataset(file_path_hist, file_path_fut, hist_key, fut_key,num_streams=32, snr_range=(5.0, 20.0),
                  add_noise_flag=True, seed=42, norm_stats=None, fit_norm=False, shuffle_pairs=True):
    """
    Build ONE dataset from .mat files.

    Expected input:
        H_hist: [Scenes, Users, P, Pol, Mv, Mh]
        H_fut : [Scenes, Users, L, Pol, Mv, Mh]

    Returns:
        dataset, norm_stats, meta
    """
    np.random.seed(seed)

    H_hist = hdf5storage.loadmat(file_path_hist)[hist_key]
    H_fut = hdf5storage.loadmat(file_path_fut)[fut_key]

    # [Scenes, Users, T, Pol, Mv, Mh] -> [(Scenes*Users), T, (Pol*Mv*Mh)]
    H_hist = rearrange(H_hist, 's u t p mv mh -> (s u) t (p mv mh)')
    H_fut = rearrange(H_fut, 's u t p mv mh -> (s u) t (p mv mh)')

    B, prev_len, mul = H_hist.shape
    _, pred_len, mul2 = H_fut.shape

    assert mul == mul2, f"Mismatch in channel dimensions: {mul} vs {mul2}"
    assert mul % num_streams == 0, f"mul={mul} must be divisible by num_streams={num_streams}"

    if shuffle_pairs:
        dt = np.concatenate((H_hist, H_fut), axis=1)
        np.random.shuffle(dt)
        H_hist = dt[:, :prev_len, ...]
        H_fut = dt[:, prev_len:, ...]

    if add_noise_flag:
        for i in range(B):
            snr_i = random.rand() * (snr_range[1] - snr_range[0]) + snr_range[0]
            H_hist[i, ...] = noise(H_hist[i, ...], snr_i)
            # H_fut stays clean

    # complex -> real
    H_hist = LoadBatch_narrowband(H_hist, num=num_streams)   # [N, P, 2]
    H_fut = LoadBatch_narrowband(H_fut, num=num_streams)     # [N, L, 2]

    if fit_norm:
        norm_stats = compute_norm_stats(H_hist, H_fut)

    if norm_stats is None:
        raise ValueError("norm_stats must be provided unless fit_norm=True")

    H_hist, H_fut = apply_normalization(H_hist, H_fut, norm_stats)

    meta = {
        "channel_normalization": "train-set feature-wise mean/std on real-imag representation",
        "prev_len": prev_len,
        "pred_len": pred_len,
        "mul": mul,
        "num_streams": num_streams,
    }

    dataset = Dataset_Pro(prev=H_hist, fut=H_fut)
    return dataset, norm_stats, meta