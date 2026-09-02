import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from .data_LLM4CP import build_dataset
from .GPT4CP import Model
from metrics import NMSELoss
import json
from torch.utils.data import DataLoader

epochs = 400
batch_size = 1024
num_workers = 2
resume_training = False

def save_best_checkpoint(model, save_path):
    torch.save(model.state_dict(), save_path)

def save_norm_stats(norm_stats, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=4)

def plot_losses(train_losses, val_losses):
    epochs_axis = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs_axis, train_losses, label="Train Loss")
    plt.plot(epochs_axis, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

def train(model, training_data_loader, validate_data_loader, optimizer, criterion, device, save_path):
    best_loss = float("inf")
    patience = 40
    epoch_without_improvment = 0
    train_losses = []
    val_losses = []

    print("Start training...")
    for epoch in range(epochs):
        print(f"starting epoch {epoch + 1}/{epochs}")
        epoch_train_loss, epoch_val_loss = [], []

        # ===== TRAIN =====
        model.train()
        for iteration, batch in enumerate(training_data_loader, 1):

            pred_t = batch[0].to(device, non_blocking=True)
            prev = batch[1].to(device, non_blocking=True)
            optimizer.zero_grad()
            pred_m = model(prev, None, None, None)
            loss = criterion(pred_m, pred_t)
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss = np.nanmean(epoch_train_loss)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{epochs} train loss: {train_loss:.7f}")

        # ===== VALIDATION =====
        model.eval()
        with torch.no_grad():

            for batch in validate_data_loader:
                pred_t = batch[0].to(device, non_blocking=True)
                prev = batch[1].to(device, non_blocking=True)
                pred_m = model(prev, None, None, None)
                loss = criterion(pred_m, pred_t)
                epoch_val_loss.append(loss.item())

        val_loss = np.nanmean(epoch_val_loss)
        val_losses.append(val_loss)

        print(f"validation loss: {val_loss:.7f}")
        if val_loss < best_loss:
            best_loss = val_loss
            save_best_checkpoint(model, save_path)
            print(f"best model saved with val loss {best_loss:.7f}")
            epoch_without_improvment = 0

        else:
            epoch_without_improvment += 1

        if epoch_without_improvment >= patience:
            print(f"Early stopping at epoch {epoch + 1}. "
                  f"No validation improvement for {patience} epochs.")
            break

    return train_losses, val_losses

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    scenario = "RMa"  # "UMa" or "RMa"
    os.makedirs("Weights", exist_ok=True)

    save_path = f"Weights/LLM4CP_{scenario}.pth"
    norm_stats_path = f"Weights/LLM4CP_{scenario}_norm_stats.json"

    train_hist_path = f"data/{scenario}/train_val/H_hist_train_noisy.mat"
    train_fut_path = f"data/{scenario}/train_val/H_fut_train.mat"
    val_hist_path = f"data/{scenario}/train_val/H_hist_val_noisy.mat"
    val_fut_path = f"data/{scenario}/train_val/H_fut_val.mat"
    resume_path = save_path
    # Build train, validation datasets
    train_set, norm_stats, train_meta = build_dataset(file_path_hist=train_hist_path,file_path_fut=train_fut_path,hist_key="H_hist_train",fut_key="H_fut_train",num_streams=32,
        norm_stats=None,fit_norm=True,shuffle_pairs=True)

    save_norm_stats(norm_stats, norm_stats_path)

    validate_set, _, val_meta = build_dataset(file_path_hist=val_hist_path,file_path_fut=val_fut_path,hist_key="H_hist_val",fut_key="H_fut_val",num_streams=32,
                                              norm_stats=norm_stats,fit_norm=False,shuffle_pairs=False)

    print("Train meta:", train_meta)
    print("Val meta:", val_meta)
    print("Train set size:", len(train_set))
    print("Validation set size:", len(validate_set))

    # Peek one sample to verify dimensions
    sample_pred, sample_prev = train_set[0]
    print("One train sample shapes:")
    print("pred:", sample_pred.shape)   # expected [L, 2]
    print("prev:", sample_prev.shape)   # expected [P, 2]

    model = Model(pred_len=4, prev_len=16).to(device)

    if resume_training:
        if os.path.exists(resume_path):
            print(f"\nLoading checkpoint from: {resume_path}")
            state_dict = torch.load(resume_path,map_location=device,weights_only=True)
            model.load_state_dict(state_dict)
            print("Checkpoint loaded successfully.\n")
        else:
            print(f"\nCheckpoint not found: {resume_path}")
            print("Training will start from scratch.\n")
    else:
        print("\nTraining from scratch.\n")

    total = sum(param.nelement() for param in model.parameters())
    print("Number of parameter: %.5fM" % (total / 1e6))

    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

    training_data_loader = DataLoader(train_set,batch_size=batch_size,
        shuffle=True,num_workers=num_workers,pin_memory=True,persistent_workers=(num_workers > 0),drop_last=True)

    validate_data_loader = DataLoader(validate_set,batch_size=batch_size,
        shuffle=False,num_workers=num_workers,pin_memory=True,persistent_workers=(num_workers > 0),drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001)
    criterion = NMSELoss().to(device)

    train_losses, val_losses = train(model=model,training_data_loader=training_data_loader,validate_data_loader=validate_data_loader,
        optimizer=optimizer,criterion=criterion,device=device,save_path=save_path)

    plot_losses(train_losses,val_losses)