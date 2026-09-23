import sys
import os
import json
from pathlib import Path
import pandas as pd
import torch
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.AASIST import Model
from modules.detector_ai_voice.infer import NUM_EVAL_SAMPLES

def main():
    with open(REPO_ROOT / 'config' / 'AASIST.conf', 'r') as f:
        cfg = json.load(f)

    model = Model(cfg['model_config'])
    ckpt = torch.load(REPO_ROOT / 'models' / 'best_AASIST_VoxGuard_zero_pad.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    df = pd.read_csv(REPO_ROOT / 'data' / 'subset_metadata.csv')
    la_df = df[df['dataset'] == 'LA']

    bonafides = la_df[la_df['label'] == 'bonafide'].head(5)
    spoofs = la_df[la_df['label'] == 'spoof'].head(5)

    def get_logits(fp):
        data, sr = sf.read(fp)
        if data.ndim > 1: data = data.mean(axis=-1)
        if len(data) < NUM_EVAL_SAMPLES:
            data = torch.from_numpy(data).float()
            data = torch.cat([data, torch.zeros(NUM_EVAL_SAMPLES - len(data))])
        else:
            data = torch.from_numpy(data[:NUM_EVAL_SAMPLES]).float()
        with torch.no_grad():
            _, logits = model(data.unsqueeze(0))
        return logits[0, 0].item(), logits[0, 1].item()

    print("--- BONAFIDE SAMPLES ---")
    for idx, row in bonafides.iterrows():
        fp = REPO_ROOT / row['subset_filepath']
        if fp.exists():
            l0, l1 = get_logits(fp)
            diff = l0 - l1
            print(f"{fp.name:30s} | Logit0: {l0:6.2f} | Logit1: {l1:6.2f} | Diff (l0-l1): {diff:6.2f}")

    print("\n--- SPOOF SAMPLES ---")
    for idx, row in spoofs.iterrows():
        fp = REPO_ROOT / row['subset_filepath']
        if fp.exists():
            l0, l1 = get_logits(fp)
            diff = l0 - l1
            print(f"{fp.name:30s} | Logit0: {l0:6.2f} | Logit1: {l1:6.2f} | Diff (l0-l1): {diff:6.2f}")

if __name__ == "__main__":
    main()
