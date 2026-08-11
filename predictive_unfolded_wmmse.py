import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

class Predictive_UnfoldedWMMSE(nn.Module):
    def __init__(self, L, N, K, noise_power, Gamma, delta=1):
        super().__init__()
        self.NPGRs = nn.ModuleList([NPGR(delta=delta) for l in range(L)])
        self.Gamma = torch.nn.Parameter(Gamma[:L].clone())# [L,K]
        self.L = L
        self.N = N
        self.K = K
        self.noise_power = noise_power
        self.log2 = torch.log(torch.tensor(2.0, dtype=DTYPE, device=device))

    def build_initial_precoder(self, channel, total_power):
        """
        channel : [B,N,2M,2]
        Returns:
            V0 : [B,N,2M,1]
        """
        h_vec = channel[..., 0].unsqueeze(-1)  # [B,N,2M,1]
        power = torch.sum(h_vec ** 2, dim=(1, 2, 3), keepdim=True) + 1e-12
        V0 = h_vec / torch.sqrt(power) * torch.sqrt(torch.tensor(total_power, device=h_vec.device, dtype=h_vec.dtype))
        return V0

    def project_power(self, V, total_power):
        B = V.shape[0]
        norms_sq = torch.sum(V ** 2, dim=(1, 2, 3))
        scale_proj = torch.ones_like(norms_sq)
        mask = norms_sq > total_power
        scale_proj[mask] = (total_power ** 0.5) / torch.sqrt(norms_sq[mask]) # sqrt(P) / ||V||
        return V * scale_proj.view(B, 1, 1, 1) # V * sqrt(P) / ||V||

    def compute_u(self,h,V):
        idx = torch.arange(self.N , device=h.device)
        h_T = h.transpose(-1, -2)
        hV = torch.matmul(h_T.unsqueeze(2), V.unsqueeze(1))  # [B,N,N,2,1]
        p = torch.sum(hV ** 2, dim=(-2, -1))  # [B,N,N]
        total_power_tensor = p.sum(dim=2) + self.noise_power
        u = hV[:, idx, idx] / total_power_tensor.unsqueeze(-1).unsqueeze(-1)  # [B, N, 2, 1]
        return u

    def compute_w(self,h,V):
        idx = torch.arange(self.N , device=h.device)
        h_T = h.transpose(-1, -2)
        hV = torch.matmul(h_T.unsqueeze(2), V.unsqueeze(1))  # [B,N,N,2,1]
        p = torch.sum(hV ** 2, dim=(-2, -1))  # [B,N,N]
        p_ii = p[:, idx, idx]
        total_power_tensor = p.sum(dim=2) + self.noise_power
        inter_power = total_power_tensor - p_ii
        w = (total_power_tensor / (inter_power + 1e-12))
        return w

    def compute_u_w_seq(self, h_seq, V):
        # h_seq: [B,T,N,2M,2]
        # V:     [B,N,2M,1]

        idx = torch.arange(self.N, device=h_seq.device)
        h_T = h_seq.transpose(-1, -2)  # [B,T,N,2,2M]
        hT_exp = h_T.unsqueeze(3)  # [B,T,N,1,2,2M]
        V_exp = V.unsqueeze(1).unsqueeze(2)  # [B,1,1,N,2M,1]
        hV = torch.matmul(hT_exp, V_exp)  # [B,T,N,N,2,1]
        p = torch.sum(hV ** 2, dim=(-2, -1))  # [B,T,N,N]
        p_ii = p[:, :, idx, idx]  # [B,T,N]
        total = p.sum(dim=3) + self.noise_power  # [B,T,N]
        inter = total - p_ii
        u_seq = hV[:, :, idx, idx] / total.unsqueeze(-1).unsqueeze(-1)
        w_seq = total / (inter + 1e-12)
        return u_seq, w_seq

    def compute_gradient(self,
                 w,  # [B, N]
                 user_weights,  # [B, N, 1]
                 u,  # [B, N, 2, 1]
                 channel,  # [B, N, 2M, 2]
                 V): # [B, N, 2M, 1]

        B, N, twoM, _ = channel.shape
        h = channel                      # [B, N, 2M, 2]
        h_T = h.transpose(-1, -2)        # [B, N, 2, 2M]
        hhT = torch.matmul(h, h_T)       # h_i h_i^T → [B,N,2M,2M] : real-valued equivalent of h_i h_i^H
        alpha = user_weights.squeeze(-1)  #[B,N] αi i=1..N
        u_norm_sq = torch.sum(u ** 2, dim=(-2, -1))  # [B,N]   ||ui||^2
        scale = (w * alpha * u_norm_sq).unsqueeze(-1).unsqueeze(-1)  # [B,N,1,1]
        A = torch.sum(scale * hhT, dim=1)   # Σ_i αi * wi * ||ui||^2 * hi*hi^T → [B,2M,2M]

        h_u = torch.matmul(h, u)  # [B,N,2M,1]
        grad_1 = -2.0 * w.unsqueeze(-1).unsqueeze(-1) * alpha.unsqueeze(-1).unsqueeze(-1) * h_u # grad_1 = -2*α*w*hu [B,N,2M,1]
        sum_grad_exp = A.unsqueeze(1)  # [B,1,2M,2M]
        A_V = torch.matmul(sum_grad_exp, V)  # [B,N,2M,1]
        gradient = grad_1 + 2.0 * A_V  # [B,N,2M,1]
        return gradient

    def compute_gradient_seq(self, w_seq, user_weights, u_seq, h_seq, V):
        # w_seq:       [B,T,N]
        # user_weights:[B,N,1]
        # u_seq:       [B,T,N,2,1]
        # h_seq:       [B,T,N,2M,2]
        # Vin:         [B,N,2M,1]

        h_T = h_seq.transpose(-1, -2)  # [B,T,N,2,2M]
        hhT = torch.matmul(h_seq, h_T)  # [B,T,N,2M,2M]
        alpha = user_weights.squeeze(-1)  # [B,N]
        u_norm_sq = torch.sum(u_seq ** 2, dim=(-2, -1))  # [B,T,N]
        scale = (w_seq * alpha.unsqueeze(1) * u_norm_sq).unsqueeze(-1).unsqueeze(-1)  # [B,T,N,1,1]
        A = torch.sum(scale * hhT, dim=2)  # [B,T,2M,2M]
        h_u = torch.matmul(h_seq, u_seq)  # [B,T,N,2M,1]
        grad_1 = -2.0 * w_seq.unsqueeze(-1).unsqueeze(-1) * alpha.unsqueeze(1).unsqueeze(-1).unsqueeze(-1) * h_u  # [B,T,N,2M,1]
        A_V = torch.matmul(
            A.unsqueeze(2),  # [B,T,1,2M,2M]
            V.unsqueeze(1)  # [B,1,N,2M,1]
        )  # [B,T,N,2M,1]

        gradient_seq = grad_1 + 2.0 * A_V
        return gradient_seq

    def PGD_step(self,
                 gamma_lk,
                 gradient,  # [B, N, 2M, 1]
                 Vin,  # [B, N, 2M, 1]
                 total_power):

        Vout_temp = Vin - gradient * gamma_lk
        return self.project_power(Vout_temp, total_power)

    def forward(self, mob_features, channel_hist_seq, user_weights , total_power, stage, num_layers_eval=None):

        if num_layers_eval is None:
            num_layers_eval = self.L

        if not 1 <= num_layers_eval <= self.L:
            raise ValueError(f"num_layers_eval must be between 1 and {self.L}")

        channel_n = channel_hist_seq[:,-1]
        B, N, twoM, _ = channel_n.shape
        V = self.build_initial_precoder(channel_n, total_power) #V(0,0)

        for l in range(num_layers_eval):
            if stage == 1:
                w = self.compute_w(channel_n, V)
                u = self.compute_u(channel_n, V)
            elif stage == 2:
                NPGR_l = self.NPGRs[l]
                u_seq, w_seq = self.compute_u_w_seq(channel_hist_seq, V)  # [B,T,N,2,1] u{n-4},...,u{n} and [B,T,N] w{n-4},...,w{n}
            gamma_l = self.Gamma[l,:] #[K]

            # ----- K PGD steps -----
            for k in range(self.K):
                # build gradient sequence with the current V:
                if stage == 1:
                    grad = self.compute_gradient(w, user_weights, u, channel_n, V)
                elif stage == 2:
                    grad_seq = self.compute_gradient_seq(w_seq, user_weights, u_seq, channel_hist_seq, V)
                    grad = NPGR_l(mob_features,channel_hist_seq,V,grad_seq,u_seq,w_seq)
                #PGD step
                V = self.PGD_step(gamma_l[k] , grad, V, total_power)

        return V

class NPGR(nn.Module):
    def __init__(self,hidden=32, T=5, mob_channels=6,uw_channels=6,delta=1):
        super().__init__()
        self.hidden = hidden
        self.T = T
        self.mob_channels = mob_channels
        self.uw_channels = uw_channels
        self.delta = delta
        drop_p = 0.05

        # Mobility branch
        # input: [B, 6, N, T-1]
        self.mob_embed = nn.Sequential(nn.Conv2d(in_channels=6,out_channels=mob_channels,kernel_size=5,padding=2),nn.ReLU())
        self.mob_reduce = nn.Conv2d(in_channels=mob_channels,out_channels=mob_channels,kernel_size=(1, T - 1),stride=1,padding=0)

        # u,w branch
        # u_seq: [B,T,N,2,1] , w_seq: [B,T,N] , after rearranging: uw_seq: [B,3,N,T]
        # channels = Re(u), Im(u), w
        self.uw_embed = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=uw_channels,kernel_size=3,padding=1),nn.ReLU())
        self.uw_reduce = nn.Conv2d(in_channels=uw_channels,out_channels=uw_channels,kernel_size=(1, T),stride=1,padding=0)

        # H + dH + dg + V + grad_seq + mobility embedding + uw embedding
        in_ch = 4 + T + mob_channels + uw_channels
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_ch,out_channels=hidden,kernel_size=5,padding=2),nn.ReLU(),
            nn.Dropout2d(drop_p),nn.Conv2d(in_channels=hidden,out_channels=1,kernel_size=5,padding=2))

    def forward(self,mob_features,channel_hist_seq,V,grad_seq,u_seq,w_seq):
        """
        mob_features     : [B,6,N,T-1]
        channel_hist_seq : [B,T,N,2M,2]
        V                : [B,N,2M,1]
        grad_seq         : [B,T,N,2M,1]
        u_seq            : [B,T,N,2,1]
        w_seq            : [B,T,N]
        """
        B, T, N, twoM, _ = channel_hist_seq.shape
        # --------------------------
        # Main spatial inputs
        # --------------------------
        H = channel_hist_seq[:, -1, ..., 0].unsqueeze(1)   # [B,1,N,2M]
        dH = (channel_hist_seq[:, -1, ..., 0] - channel_hist_seq[:, -1 - self.delta, ..., 0]).unsqueeze(1)  # [B,1,N,2M] Hn - H(n-delta)
        grad_map = grad_seq.squeeze(-1)  # [B,T,N,2M]
        dg = (grad_map[:, -1] - grad_map[:, -1 - self.delta]).unsqueeze(1)  # [B,1,N,2M] gn - g(n-delta)
        V_map = V.squeeze(-1).unsqueeze(1) # [B,1,N,2M]

        # --------------------------
        # Mobility branch
        # --------------------------
        mob_z = self.mob_embed(mob_features)  # [B,mob_channels,N,T-1]
        mob_z = self.mob_reduce(mob_z) # [B,mob_channels,N,1]
        mob_z = mob_z.expand(-1, -1, -1, twoM)   # [B,mob_channels,N,2M]

        # --------------------------
        # u,w temporal branch
        # --------------------------
        # u_seq:
        u_map = u_seq.squeeze(-1) # [B,T,N,2,1] -> [B,T,N,2]
        u_map = u_map.permute(0, 3, 2, 1) # [B,T,N,2] -> [B,2,N,T]
        # w_seq:
        w_map = w_seq.permute(0, 2, 1).unsqueeze(1) # [B,T,N] -> [B,1,N,T]
        uw_in = torch.cat([u_map, w_map],dim=1)     # [B,3,N,T]
        uw_z = self.uw_embed(uw_in) # [B,uw_channels,N,T]
        uw_z = self.uw_reduce(uw_z)     # [B,uw_channels,N,1]
        uw_z = uw_z.expand(-1, -1, -1, twoM)  # [B,uw_channels,N,2M]

        # --------------------------
        # Global fusion
        # --------------------------
        cnn_in = torch.cat([H,dH,grad_map,dg,V_map,mob_z,uw_z],dim=1)
        grad_pred = self.conv(cnn_in)              # [B,1,N,2M]
        grad_pred = grad_pred.permute(0, 2, 3, 1)  # [B,N,2M,1]
        return grad_pred
