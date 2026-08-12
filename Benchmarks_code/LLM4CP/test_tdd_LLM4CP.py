import torch
import numpy as np
import hdf5storage
from torch.utils.data import DataLoader
from einops import rearrange
import json
from .data_LLM4CP import LoadBatch_narrowband, Dataset_Pro, apply_normalization
from metrics import NMSELoss
from .GPT4CP import Model

def build_test_dataset(file_path_hist, file_path_fut, hist_key="H_hist_test", fut_key="H_fut_test", num_streams=32, norm_stats = None):
    """
    Build test dataset from MATLAB files.

    Expected input shapes:
        H_hist: [Scenes, Users, P, Pol, Mv, Mh]
        H_fut : [Scenes, Users, L, Pol, Mv, Mh]

    Output:
        test_set: Dataset_Pro with:
            prev: [B_total * num_streams, P, 2]
            fut : [B_total * num_streams, L, 2]
    """

    H_hist = hdf5storage.loadmat(file_path_hist)[hist_key]
    H_fut = hdf5storage.loadmat(file_path_fut)[fut_key]

    # [Scenes, Users, T, Pol, Mv, Mh] -> [(Scenes*Users), T, (Pol*Mv*Mh)]
    H_hist = rearrange(H_hist, "s u t p mv mh -> (s u) t (p mv mh)")
    H_fut = rearrange(H_fut, "s u t p mv mh -> (s u) t (p mv mh)")

    B, prev_len, mul = H_hist.shape
    B2, pred_len, mul2 = H_fut.shape

    assert B == B2, f"Mismatch in number of samples: {B} vs {B2}"
    assert mul == mul2, f"Mismatch in channel dimension: {mul} vs {mul2}"
    assert mul % num_streams == 0, f"mul={mul} must be divisible by num_streams={num_streams}"

    # Complex -> real, stream-wise split
    H_hist = LoadBatch_narrowband(H_hist, num=num_streams)   # [B*num_streams, P, 2]
    H_fut = LoadBatch_narrowband(H_fut, num=num_streams)     # [B*num_streams, L, 2]

    H_hist, H_fut = apply_normalization(H_hist, H_fut, norm_stats)

    test_set = Dataset_Pro(prev=H_hist, fut=H_fut)

    meta = {
        "num_scene_user_pairs": B,
        "prev_len": prev_len,
        "pred_len": pred_len,
        "mul": mul,
        "num_streams": num_streams,
        "total_stream_samples": len(test_set),
    }

    return test_set, meta


def load_model_checkpoint(model, ckpt_path, device):
    """
    Loads either:
    1. a state_dict
    2. a checkpoint dict containing model_state_dict / state_dict
    3. a full saved model
    """
    print(f"\nLoading model from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Case 1: full model object
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
        print("Loaded full model object.")
        return model

    # Case 2: checkpoint dict or pure state_dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        print("Loaded state_dict.")
        return model

    raise ValueError("Unsupported checkpoint format.")


def evaluate_llm4cp(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for fut, prev in data_loader:
            prev = prev.to(device)
            fut = fut.to(device)

            pred = model(prev, None, None, None)
            loss = criterion(pred, fut)

            bsz = fut.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz

    return total_loss / total_samples


def evaluate_no_prediction(data_loader, criterion, device):
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for fut, prev in data_loader:
            prev = prev.to(device)   # [B, P, 2]
            fut = fut.to(device)     # [B, L, 2]

            # Repeat last observed channel over the prediction horizon
            pred = prev[:, [-1], :].repeat(1, fut.shape[1], 1)
            loss = criterion(pred, fut)

            bsz = fut.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz

    return total_loss / total_samples


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # ========================= PATHS =========================
    test_hist_path = "data/UMa/test/H_hist_test_noisy.mat"
    test_fut_path = "data/UMa/test/H_fut_test.mat"
    norm_stats_path = "Weights/LLM4CP_UMa_norm_stats.json"

    hist_key = "H_hist_test"
    fut_key = "H_fut_test"

    model_path = "Weights/LLM4CP_original.pth"

    batch_size = 512
    num_workers = 0
    num_streams = 32

    with open(norm_stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    norm_stats = {
        "mean": torch.tensor(stats["mean"], dtype=torch.float32),
        "std": torch.tensor(stats["std"], dtype=torch.float32),
    }
    # ========================================================

    test_set, meta = build_test_dataset(file_path_hist=test_hist_path,file_path_fut=test_fut_path,hist_key=hist_key,fut_key=fut_key,num_streams=num_streams, norm_stats = norm_stats)

    print("\nTest meta:")
    for k, v in meta.items():
        print(f"{k}: {v}")

    sample_fut, sample_prev = test_set[0]
    print("\nOne test sample shapes:")
    print("fut :", sample_fut.shape)    # expected [L, 2]
    print("prev:", sample_prev.shape)   # expected [P, 2]

    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True, drop_last=False)

    criterion = NMSELoss().to(device)

    # ===================== LLM4CP =====================
    model = Model(pred_len=meta["pred_len"], prev_len=meta["prev_len"]).to(device)
    model = load_model_checkpoint(model, model_path, device)

    nmse_llm4cp = evaluate_llm4cp(model=model,data_loader=test_loader,criterion=criterion,device=device)

    # ================= No Prediction ==================
    nmse_np = evaluate_no_prediction(data_loader=test_loader,criterion=criterion,device=device)

    # ===================== Print ======================
    print("\n==================== TEST RESULTS ====================")
    print(f"LLM4CP        NMSE: {nmse_llm4cp:.7f}")
    print(f"No Prediction NMSE: {nmse_np:.7f}")
