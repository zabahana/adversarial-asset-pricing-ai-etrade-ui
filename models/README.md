# Model Artifacts

Place the trained checkpoints required by the Lightning application in the following locations:

- `models/dqn/latest.ckpt`
- `models/mha_dqn/clean.ckpt`
- `models/mha_dqn/adversarial.ckpt`

These files should be serialized PyTorch modules (e.g., created with `torch.save`). The application will raise a `FileNotFoundError` if any checkpoint is missing.
