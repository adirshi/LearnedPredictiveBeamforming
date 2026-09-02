import torch

@torch.no_grad()
def predict_future_with_kf(H_hist_seq, measurement_noise_var, ue_speed, delta=1,
        carrier_frequency=2.4e9, sampling_interval=0.5e-3, channel_variance=1.0):
    """
    Vectorized KF prediction for ONE complete channel realization.
    All N*M channel coefficients are processed in parallel.
    H_hist_seq: [N,T,M] complex
    measurement_noise_var: [N]
    ue_speed: [N]
    Returns:H_future_pred: [N,M] complex
    """

    N, T, M = H_hist_seq.shape
    # =========================================================
    # Parameters for each UE
    # =========================================================
    f_D = ue_speed * carrier_frequency / 3e8
    x = 2.0 * torch.pi * f_D * sampling_interval

    # Jakes temporal correlation
    a = torch.special.bessel_j0(x)

    # Process noise
    Q = channel_variance * (1.0 - a.abs() ** 2)
    Q = torch.clamp(Q,min=1e-12)

    # Measurement noise
    R = measurement_noise_var

    # ---------------------------------------------------------
    # Add coefficient dimension:
    # Broadcasting then applies each UE parameter to all M
    # coefficients simultaneously.
    # ---------------------------------------------------------
    a = a[:, None]       # [N,1]
    Q = Q[:, None]       # [N,1]
    R = R[:, None]       # [N,1]

    # =========================================================
    # KF initialization
    # =========================================================

    # First historical CSI sample
    h_hat = H_hist_seq[:, 0, :]

    # Initial estimation-error variance
    P = torch.full((N, M),channel_variance,dtype=torch.float32,device=H_hist_seq.device)

    # =========================================================
    # Kalman filtering
    # Only loop remaining is over TIME.
    # At each t, all N*M coefficients are processed in parallel.
    # =========================================================
    for t in range(1, T):
        # -----------------------------------------------------
        # Prediction
        # -----------------------------------------------------
        h_pred = a * h_hat
        P_pred = (a.abs() ** 2) * P + Q

        # -----------------------------------------------------
        # Innovation
        # -----------------------------------------------------
        innovation = (H_hist_seq[:, t, :] - h_pred)
        innovation_var = P_pred + R

        # -----------------------------------------------------
        # Kalman gain
        # -----------------------------------------------------
        kalman_gain = P_pred / innovation_var

        # -----------------------------------------------------
        # Measurement update
        # -----------------------------------------------------
        h_hat = h_pred + kalman_gain * innovation
        P = (1.0 - kalman_gain)* P_pred

    # =========================================================
    # Delta-step prediction
    # =========================================================
    H_future_pred = (a ** delta) * h_hat
    return H_future_pred

def kalman_predict_coefficient(noisy_sequence,measurement_noise_var,process_noise_var,transition_coefficient,delta=1,initial_channel_variance=1.0):
    """
    Kalman prediction for one complex channel coefficient.
    State model:
        h(t+1) = a * h(t) + q(t)
    Measurement model:
        y(t) = h(t) + e(t)
    where:
        q_t ~ CN(0, Q)
        e_t ~ CN(0, R)

    Inputs
    ------
    noisy_sequence: Noisy historical CSI samples. Shape: [T] [h_hat_(t-T+1), ..., h_hat_t]
    measurement_noise_var: Measurement-noise variance R.
    process_noise_var: Process-noise variance Q.
    transition_coefficient: AR(1) transition coefficient a.

    Returns
    -------
    h_future_pred: Complex delta-step channel prediction h_hat_(t+delta|t).
    """

    T = len(noisy_sequence)
    a = transition_coefficient
    Q = process_noise_var
    R = measurement_noise_var
    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------
    # Use the first noisy CSI sample as the initial state estimate
    h_hat = complex(noisy_sequence[0])

    # Initial state-estimation error variance
    P = float(initial_channel_variance)

    # ---------------------------------------------------------
    # Kalman filtering over the remaining historical samples
    # ---------------------------------------------------------
    for t in range(1, T):

        # ----- Prediction -----
        # h_hat_(t|t-1) = a * h_hat_(t-1|t-1)
        h_pred = a * h_hat

        # P_(t|t-1) = |a|^2 P_(t-1|t-1) + Q
        P_pred = (abs(a) ** 2) * P + Q

        # ----- Innovation -----
        # innovation = y(t) - h_hat_(t|t-1)
        innovation = noisy_sequence[t] - h_pred

        # Innovation variance:
        # S(t) = P_(t|t-1) + R
        innovation_var = P_pred + R

        # ----- Kalman gain -----
        # K(t) = P_(t|t-1) / S(t)
        kalman_gain = P_pred / innovation_var

        # ----- Measurement update -----
        # h_hat_(t|t) = h_hat_(t|t-1) + K(t) * innovation
        h_hat = h_pred + kalman_gain  * innovation

        # ----- Error covariance update -----
        # P_(t|t) = (1 - K_t) P_(t|t-1)
        P = (1.0 - kalman_gain) * P_pred

    # ---------------------------------------------------------
    # Delta-step future prediction
    # No future measurements are available, so after processing the final historical CSI sample we perform prediction only:
    # h_hat_(t+delta|t) = a^delta * h_hat_(t|t)
    # ---------------------------------------------------------
    h_future_pred = (a ** delta) * h_hat
    return h_future_pred