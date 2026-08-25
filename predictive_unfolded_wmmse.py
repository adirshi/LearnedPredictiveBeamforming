import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

class Predictive_UnfoldedWMMSE(nn.Module):
    """
    Predictive unfolded WMMSE with neural predictive gradient refinement (NPGR).

    Stage 1:
        Uses the analytical current-channel gradient in each PGD step.

    Stage 2:
        Uses cached historical quantities together with the current quantities and mobility features
        to generate a predictive gradient using the NPGR.
    """

    def __init__(self, L, N, K, noise_power, Gamma, delta=1):
        super().__init__()

        self.L = L
        self.N = N
        self.K = K
        self.noise_power = noise_power
        # One NPGR per unfolded WMMSE iteration
        self.NPGRs = nn.ModuleList([NPGR(delta=delta) for _ in range(L)])
        # Trainable PGD step sizes: [L, K]
        self.Gamma = torch.nn.Parameter(Gamma[:L].clone())

    def initialize_precoder(self, channel, total_power):
        """
            Build the initial beamformer from the current CSI.
            Args: channel: [B, N, 2M, 2]
            Returns: V: [B, N, 2M, 1]
        """

        h_vec = channel[..., 0].unsqueeze(-1)  # [B,N,2M,1]
        power = torch.sum(h_vec ** 2, dim=(1, 2, 3), keepdim=True) + 1e-12
        V0 = h_vec / torch.sqrt(power) * torch.sqrt(torch.tensor(total_power, device=h_vec.device, dtype=h_vec.dtype))
        return V0

    def project_power(self, V, total_power):
        """
           Project the beamformer onto the transmit power constraint.
           Args: V: [B, N, 2M, 1]
        """
        B = V.shape[0]
        norms_sq = torch.sum(V ** 2, dim=(1, 2, 3))
        scale_proj = torch.ones_like(norms_sq)
        mask = norms_sq > total_power
        scale_proj[mask] = (total_power ** 0.5) / torch.sqrt(norms_sq[mask]) # sqrt(P) / ||V||
        return V * scale_proj.view(B, 1, 1, 1) # V * sqrt(P) / ||V||

    def compute_u_w(self, h, V):
        """
        Compute the WMMSE receiver variables and MSE weights.
        Args: h: [B, N, 2M, 2]
              V: [B, N, 2M, 1]

        Returns: u: [B, N, 2, 1]
                 w: [B, N]
        """
        idx = torch.arange(self.N, device=h.device)
        h_T = h.transpose(-1, -2)  # [B,N,2,2M]

        # All user-channel / beamformer products
        hV = torch.matmul(h_T.unsqueeze(2),V.unsqueeze(1))  # [B,N,N,2,1]

        # |h_i^H v_j|^2 for all i,j
        p = torch.sum(hV ** 2,dim=(-2, -1))  # [B,N,N]

        # Desired signal power
        p_ii = p[:, idx, idx]  # [B,N]

        # Total received power = signal + interference + noise
        total_power_tensor = p.sum(dim=2) + self.noise_power # [B,N]

        # WMMSE receiver
        u = hV[:, idx, idx] / total_power_tensor.unsqueeze(-1).unsqueeze(-1) # [B,N,2,1]

        # Interference + noise
        inter_power = total_power_tensor - p_ii

        # WMMSE MSE weight
        w = total_power_tensor / (inter_power + 1e-12)  # [B,N]
        return u, w

    def compute_gradient(self, w, user_weights, u, channel, V):

        """
           Compute the analytical WMMSE gradient.
           Args:
               w:            [B, N]
               user_weights: [B, N, 1]
               u:            [B, N, 2, 1]
               channel:      [B, N, 2M, 2]
               V:            [B, N, 2M, 1]

           Returns:
               gradient:     [B, N, 2M, 1]
           """

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

    def pgd_step(self, gamma_lk, gradient, Vin, total_power):
        """
             Perform one projected-gradient step.
        """
        return self.project_power(Vin - gradient * gamma_lk, total_power)

    def forward(self,H_t,mob_features,cache,total_power,user_weights, stage):

        """
               Args:
                   H_t: Current CSI, [B,N,2M,2].
                   mob_features: mobility history used by the NPGR.
                   cache: Historical quantities from previous time instances, Required only in Stage 2.

                   stage:
                       1 -> unfolded WMMSE using analytical gradients.
                       2 -> predictive unfolded WMMSE with NPGR.

               Returns:
                   V: Final beamformer.
                   current_entry: Current quantities to be cached for the next time instance.
        """

        V = self.initialize_precoder(H_t, total_power) #V(0,K)
        if stage == 2: # Historical CSI + current CSI
            channel_hist_seq = torch.cat([cache["H"], H_t.unsqueeze(1)], dim=1)  # [B,T,N,2M,2]

        current_u = []
        current_w = []
        current_grad = []
        # -----------------------------------------------------
        # L unfolded WMMSE iterations
        # -----------------------------------------------------
        for l in range(self.L):
            grad_l = []
            u_t, w_t = self.compute_u_w(H_t, V)

            if stage == 2:
                u_seq = torch.cat([cache["u"][l], u_t.unsqueeze(1)],dim=1)        # [B,T,N,2,1]
                w_seq = torch.cat([cache["w"][l], w_t.unsqueeze(1)],dim=1)        # [B,T,N]

            current_u.append(u_t)
            current_w.append(w_t)

            # -------------------------------------------------
            # K PGD steps
            # -------------------------------------------------
            for k in range(self.K):
                # Analytical current-channel gradient
                grad_t = self.compute_gradient(w_t, user_weights, u_t, H_t, V)
                if stage == 1:
                    grad_update = grad_t

                elif stage == 2:
                    # Historical analytical gradients + current analytical gradient
                    grad_seq = torch.cat([cache["grad"][l][k], grad_t.unsqueeze(1)], dim=1)  # [B,T,N,2M,1]
                    # Predictive gradient
                    grad_update = self.NPGRs[l](mob_features,channel_hist_seq,V,grad_seq,u_seq,w_seq)

                V = self.pgd_step(self.Gamma[l, k], grad_update, V, total_power)
                # Cache the analytical gradient G_t^(l,k)
                grad_l.append(grad_t)
            current_grad.append(grad_l)

        # Quantities stored for the next time instance
        current_entry = {
            "H": H_t,
            "u": current_u,
            "w": current_w,
            "grad": current_grad}
        return V, current_entry

class NPGR(nn.Module):
    """
       Neural Predictive Gradient Refinement module.
    """
    def __init__(self,hidden=32, T=5, mob_channels=6,uw_channels=6,delta=1):
        super().__init__()
        self.T = T
        self.delta = delta

        drop_p = 0.05

        # Mobility branch
        # input: [B, 6, N, T-1]
        self.mob_embed = nn.Sequential(nn.Conv2d(in_channels=6,out_channels=mob_channels,kernel_size=5,padding=2),nn.ReLU())
        self.mob_reduce = nn.Conv2d(in_channels=mob_channels,out_channels=mob_channels,kernel_size=(1, T - 1),stride=1,padding=0)

        # -----------------------------------------------------
        # WMMSE u,w temporal branch
        # u_seq: [B,T,N,2,1], w_seq: [B,T,N]
        # Combined input: [Re(u), Im(u), w] -> [B,3,N,T]
        # -----------------------------------------------------
        self.uw_embed = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=uw_channels,kernel_size=3,padding=1),nn.ReLU())
        self.uw_reduce = nn.Conv2d(in_channels=uw_channels,out_channels=uw_channels,kernel_size=(1, T),stride=1,padding=0)

        # -----------------------------------------------------
        # Main NPGR CNN
        # Inputs: H, dH, gradient sequence, dG, V,
        # mobility embedding, u/w embedding
        # -----------------------------------------------------
        in_ch = 4 + T + mob_channels + uw_channels
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_ch,out_channels=hidden,kernel_size=5,padding=2),nn.ReLU(),
            nn.Dropout2d(drop_p),nn.Conv2d(in_channels=hidden,out_channels=1,kernel_size=5,padding=2))

    def forward(self,mob_features,channel_hist_seq,V,grad_seq,u_seq,w_seq):
        """
               Args:mob_features:[B,6,N,T-1]
                    channel_hist_seq:[B,T,N,2M,2]
                    V:[B,N,2M,1]
                    grad_seq:[B,T,N,2M,1]
                    u_seq:[B,T,N,2,1]
                    w_seq:[B,T,N]

               Returns: grad_pred:[B,N,2M,1]
        """
        twoM= channel_hist_seq.shape[3]
        # -----------------------------------------------------
        # Main spatial features
        # -----------------------------------------------------
        H = channel_hist_seq[:, -1, ..., 0].unsqueeze(1)   # [B,1,N,2M]
        dH = (channel_hist_seq[:, -1, ..., 0] - channel_hist_seq[:, -1 - self.delta, ..., 0]).unsqueeze(1)  # [B,1,N,2M] Hn - H(n-delta)
        grad_map = grad_seq.squeeze(-1)  # [B,T,N,2M]
        dg = (grad_map[:, -1] - grad_map[:, -1 - self.delta]).unsqueeze(1)  # [B,1,N,2M] gn - g(n-delta)
        V_map = V.squeeze(-1).unsqueeze(1) # [B,1,N,2M]

        # -----------------------------------------------------
        # Mobility embedding
        # -----------------------------------------------------
        mob_z = self.mob_embed(mob_features)  # [B,mob_channels,N,T-1]
        mob_z = self.mob_reduce(mob_z) # [B,mob_channels,N,1]
        mob_z = mob_z.expand(-1, -1, -1, twoM)   # [B,mob_channels,N,2M]

        # -----------------------------------------------------
        # WMMSE u,w temporal embedding
        # -----------------------------------------------------
        u_map = u_seq.squeeze(-1) # [B,T,N,2,1] -> [B,T,N,2]
        u_map = u_map.permute(0, 3, 2, 1) # [B,T,N,2] -> [B,2,N,T]
        w_map = w_seq.permute(0, 2, 1).unsqueeze(1) # [B,T,N] -> [B,1,N,T]
        uw_in = torch.cat([u_map, w_map],dim=1)     # [B,3,N,T]
        uw_z = self.uw_embed(uw_in) # [B,uw_channels,N,T]
        uw_z = self.uw_reduce(uw_z)     # [B,uw_channels,N,1]
        uw_z = uw_z.expand(-1, -1, -1, twoM)  # [B,uw_channels,N,2M]

        # -----------------------------------------------------
        # Feature fusion and predictive-gradient output
        # -----------------------------------------------------
        cnn_in = torch.cat([H,dH,grad_map,dg,V_map,mob_z,uw_z],dim=1)
        grad_pred = self.conv(cnn_in)              # [B,1,N,2M]
        grad_pred = grad_pred.permute(0, 2, 3, 1)  # [B,N,2M,1]
        return grad_pred
