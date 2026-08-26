import numpy as np
import torch
from einops import rearrange
from scipy.special import j0

def predict_future_with_kf(
        H_hist_raw,
        measurement_noise_var,
        ue_speed,
        delta=1,
        T=5,
        num_coefficients=32,
        carrier_frequency=2.4e9,
        sampling_interval=0.5e-3,
        speed_of_light=3e8,
        channel_variance=1.0):
    """
    Predict H_(t+delta) using a model-based Kalman filter.
    The temporal channel evolution is modeled using a first-order
    autoregressive model:
        h(t+1) = a * h(t) + q(t)

    where the temporal correlation coefficient is obtained from the Jakes model:
        a = J0(2*pi*f_D*T_s) , f_D = v * f_c / c

    Inputs
    ------
    H_hist_raw:
        Complex historical CSI.
        Shape: [S, N, hist_len, Pol, Mv, Mh]

    measurement_noise_var:
        Measurement-noise variance R for each scene and UE.
        Shape: [S, N]

    ue_speed:
        UE speed in meters/second.
        Shape: [S, N]

    delta:
        Prediction horizon in channel samples.

    T:
        Number of historical CSI samples used by the KF.

    Returns
    -------
    H_future_pred_ri:
        Predicted future channel in real-imag representation.
        Shape: [S, N, 1, M, 2]
    """

    # ---------------------------------------------------------
    # Keep the last T historical CSI samples
    # ---------------------------------------------------------
    H_hist_seq = H_hist_raw[:, :, -T:, :, :, :]
    # [S, N, T, Pol, Mv, Mh]

    # Flatten antenna and polarization dimensions:
    # [S,N,T,Pol,Mv,Mh] -> [S,N,T,M]
    H_hist_seq = rearrange(H_hist_seq,'s n t pol mv mh -> s n t (pol mv mh)')
    S, N, _, M = H_hist_seq.shape
    assert M == num_coefficients, f"Expected {num_coefficients} channel coefficients, got {M}"
    assert measurement_noise_var.shape == (S, N), f"Expected R shape {(S, N)}, got {measurement_noise_var.shape}"
    assert ue_speed.shape == (S, N), f"Expected UE speed shape {(S, N)}, "f"got {ue_speed.shape}"

    # ---------------------------------------------------------
    # Allocate predicted future channel
    # ---------------------------------------------------------
    H_future_pred = np.zeros((S, N, M), dtype=np.complex128)

    # ---------------------------------------------------------
    # Run the KF independently for every scene, UE,
    # and channel coefficient
    # ---------------------------------------------------------
    for s in range(S):
        for n in range(N):

            # CSI measurement-noise variance for this scene and UE
            R = float(measurement_noise_var[s, n])

            # UE speed [m/s]
            v = float(ue_speed[s, n])

            # Maximum Doppler frequency for this UE
            f_D = v * carrier_frequency / speed_of_light

            a = j0(2.0 * np.pi * f_D * sampling_interval)

            # Process-noise variance.
            # For the AR(1) model:
            #     h_(t+1) = a*h_t + q_t
            # stationarity with E[|h|^2] = channel_variance gives
            #     Q = sigma_h^2 * (1 - |a|^2)
            Q = channel_variance * (1.0 - abs(a) ** 2)
            Q = max(Q, 1e-12)

            for m in range(M):
                noisy_sequence = H_hist_seq[s, n, :, m]
                # [T] complex: h_hat_(t-T+1), ..., h_hat_t

                H_future_pred[s, n, m] = kalman_predict_coefficient(
                    noisy_sequence=noisy_sequence,
                    measurement_noise_var=R,
                    process_noise_var=Q,
                    transition_coefficient=a,
                    delta=delta,
                    initial_channel_variance=channel_variance)

    # ---------------------------------------------------------
    # Complex -> real/imag representation
    # [S,N,M] -> [S,N,M,2]
    # ---------------------------------------------------------
    H_future_pred_ri = np.stack([H_future_pred.real, H_future_pred.imag],axis=-1).astype(np.float32)
    # Add future-time dimension:
    # [S,N,M,2] -> [S,N,1,M,2]
    H_future_pred_ri = H_future_pred_ri[:, :, None, :, :]
    return torch.from_numpy(H_future_pred_ri)

def kalman_predict_coefficient(
        noisy_sequence,
        measurement_noise_var,
        process_noise_var,
        transition_coefficient,
        delta=1,
        initial_channel_variance=1.0):
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
    noisy_sequence:
        Noisy historical CSI samples.
        Shape: [T] [h_hat_(t-T+1), ..., h_hat_t]

    measurement_noise_var:
        Measurement-noise variance R.

    process_noise_var:
        Process-noise variance Q.

    transition_coefficient:
        AR(1) transition coefficient a.

    delta:
        Number of future channel samples to predict.

    Returns
    -------
    h_future_pred:
        Complex delta-step channel prediction h_hat_(t+delta|t).
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