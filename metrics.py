import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
log2 = torch.log(torch.tensor(2.0, dtype=DTYPE, device=device))
def compute_WSR(noise_power, user_weights, channel, precoder):
    """
    user_weights : [1, N, 1]
    channel      : [B, N, 2M, 2]
    precoder     : [B, N, 2M, 1]
    """
    B, N, twoM, _ = channel.shape
    device = channel.device

    h = channel  # [B, N, 2M, 2]
    h_T = h.transpose(-1, -2)  # [B, N, 2, 2M]
    V = precoder  # [B, N, 2M, 1]
    hT_exp = h_T.unsqueeze(2)  # [B, N, 1, 2, 2M]
    V_exp = V.unsqueeze(1)  # [B, 1, N, 2M, 1]
    hV = torch.matmul(hT_exp, V_exp)  # [B, N, N, 2, 1]
    p = torch.sum(hV ** 2, dim=(-2, -1))  # [B, N, N]  p[b, i, j] = |h_{b,i}^T v_{b,j}|^2
    idx = torch.arange(N, device=device)
    p_ii = p[:, idx, idx]  # [B, N]
    total = p.sum(dim=2)  # [B, N]

    denom = noise_power + total - p_ii  # [B, N]
    sinr = p_ii / (denom + 1e-12)  # [B, N]
    weights = user_weights.squeeze(-1)  # [1, N]

    # sum over batch and users -> scalar (sum over B and N)
    WSR = torch.sum(weights * (torch.log1p(sinr) / log2))
    return WSR / B