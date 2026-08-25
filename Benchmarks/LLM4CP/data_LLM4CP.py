import torch.utils.data as data
import torch
import numpy as np
import hdf5storage
from einops import rearrange

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

def compute_norm_stats(H_hist, eps=1e-12):
    """
    Compute the LLM4CP channel normalization factor
    from the historical complex CSI only.

    Args:
        H_hist: complex array [B, past_len, mul]

    Returns:
        norm_stats: dictionary containing one scalar normalization factor.
    """
    scale = np.sqrt(np.std(np.abs(H_hist) ** 2))
    scale = max(float(scale), eps)

    return {"scale": scale}

def apply_normalization(H_hist, H_fut, norm_stats):
    """
    Apply the same scalar normalization factor to
    historical and future CSI.
    """
    scale = norm_stats["scale"]

    H_hist = H_hist / scale
    H_fut = H_fut / scale

    return H_hist, H_fut


def denormalize_tensor(x, norm_stats):
    """
    Undo the LLM4CP scalar channel normalization.
    """
    scale = norm_stats["scale"]
    return x * scale

def build_dataset(
    file_path_hist,
    file_path_fut,
    hist_key,
    fut_key,
    num_streams=32,
    norm_stats=None,
    fit_norm=False,
    shuffle_pairs=True):

    """
    Expected input:
        H_hist: [Scenes, Users, past_len, Pol, Mv, Mh]
        H_fut : [Scenes, Users, fut_len, Pol, Mv, Mh]

    Returns:
        dataset, norm_stats, meta
    """

    H_hist = hdf5storage.loadmat(file_path_hist)[hist_key]
    H_fut = hdf5storage.loadmat(file_path_fut)[fut_key]

    # [Scenes, Users, T, Pol, Mv, Mh]
    # -> [(Scenes*Users), T, (Pol*Mv*Mh)]
    H_hist = rearrange(H_hist,'s u t p mv mh -> (s u) t (p mv mh)')

    H_fut = rearrange(H_fut,'s u t p mv mh -> (s u) t (p mv mh)')

    _, prev_len, mul = H_hist.shape
    _, pred_len, mul2 = H_fut.shape

    assert mul == mul2
    assert mul % num_streams == 0

    if shuffle_pairs:
        dt = np.concatenate((H_hist, H_fut),axis=1)
        np.random.shuffle(dt)

        H_hist = dt[:, :prev_len, ...]
        H_fut = dt[:, prev_len:, ...]

    # ---------------------------------------------------------
    # LLM4CP channel normalization
    # ---------------------------------------------------------
    if fit_norm:
        norm_stats = compute_norm_stats(H_hist)

    if norm_stats is None:
        raise ValueError ("norm_stats must be provided unless fit_norm=True")

    H_hist, H_fut = apply_normalization(H_hist,H_fut,norm_stats)

    # ---------------------------------------------------------
    # Complex -> coefficient-wise real representation
    # ---------------------------------------------------------
    H_hist = LoadBatch_narrowband(H_hist,num=num_streams)

    H_fut = LoadBatch_narrowband(H_fut,num=num_streams)

    meta = {
        "channel_normalization":
            "LLM4CP scalar normalization based on historical channel power",
        "prev_len": prev_len,
        "pred_len": pred_len,
        "mul": mul,
        "num_streams": num_streams,
    }

    dataset = Dataset_Pro(prev=H_hist,fut=H_fut)
    return dataset, norm_stats, meta