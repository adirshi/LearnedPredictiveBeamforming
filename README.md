# Learned Predictive Beamforming

Official implementation of the research work on learned predictive beamforming for multi-user wireless communication systems.

The proposed approach combines deep unfolding with learning-based prediction to directly optimize beamforming for future channel conditions without explicitly predicting the future CSI.

## Repository Structure

- `train_predictive_model.py` – Training of the proposed predictive unfolded WMMSE model.
- `predictive_unfolded_wmmse.py` – Implementation of the predictive unfolded WMMSE architecture.
- `test_models.py` – Evaluation and comparison of the proposed method and benchmark schemes.
- `Benchmarks/` – Implementations of the considered benchmark methods.
- `generate_noisy_historical_csi.py` – Generates noisy historical CSI by adding measurement noise to clean channel realizations generated using QuaDRiGa.

## Requirements

The implementation is based on Python and PyTorch and supports CUDA-enabled GPU execution.

## Citation

Citation information will be added upon publication.
