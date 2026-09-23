# Module 2 (AI / Synthetic Voice Detector) Checkpoints

Place your trained AASIST checkpoint file here with the exact name:

```
best_model.pt
```

### Supported Formats:
The loader in `modules/detector_ai_voice/infer.py` automatically supports:
1. `{"state_dict": ..., "label_map": {"bonafide": 1, "spoof": 0}}` (recommended)
2. `{"model_state_dict": ...}`
3. Raw PyTorch `state_dict` dictionary (from `torch.save(model.state_dict(), ...)`).

*Note: `.pt` and `*.pth` files in this directory are ignored by git.*
