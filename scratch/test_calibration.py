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
    ckpt = torch.load(REPO_ROOT / 'modules' / 'detector_ai_voice' / 'checkpoints' / 'best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    df = pd.read_csv(REPO_ROOT / 'data' / 'subset_metadata.csv')
    la_df = df[df['dataset'] == 'LA']

    bonafides = la_df[la_df['label'] == 'bonafide'].head(15)
    spoofs = la_df[la_df['label'] == 'spoof'].head(15)

    test_df = pd.concat([bonafides, spoofs])

    def get_margin(filepath):
        data, sr = sf.read(filepath)
        if data.ndim > 1: data = data.mean(axis=-1)
        if len(data) < NUM_EVAL_SAMPLES:
            data = torch.from_numpy(data).float()
            data = torch.cat([data, torch.zeros(NUM_EVAL_SAMPLES - len(data))])
        else:
            data = torch.from_numpy(data[:NUM_EVAL_SAMPLES]).float()
        with torch.no_grad():
            _, logits = model(data.unsqueeze(0))
        l0 = logits[0, 0].item() # spoof_index = 0
        l1 = logits[0, 1].item()
        return l0 - l1

    margins = []
    labels = []

    for idx, row in test_df.iterrows():
        fp = REPO_ROOT / row['subset_filepath']
        if fp.exists():
            m = get_margin(fp)
            margins.append(m)
            labels.append(row['label'])

    res_df = pd.DataFrame({'label': labels, 'margin': margins})
    print("--- MARGIN SUMMARY ---")
    print(res_df.groupby('label')['margin'].describe())

    # Try different mu values
    best_mu = -2.5
    best_acc = 0.0
    for mu in [x / 10.0 for x in range(-50, 10)]:
        preds = ["spoof" if m >= mu else "bonafide" for m in margins]
        acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        if acc > best_acc:
            best_acc = acc
            best_mu = mu

    print(f"\nOptimal Calibration Center (mu): {best_mu} | Accuracy: {best_acc * 100:.1f}%")

if __name__ == "__main__":
    main()
