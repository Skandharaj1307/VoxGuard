import sys
import os
import json
from pathlib import Path
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.detector_ai_voice.infer import AIVoiceDetector

def main():
    metadata_path = REPO_ROOT / "data" / "subset_metadata.csv"
    df = pd.read_csv(metadata_path)
    
    la_df = df[df['dataset'] == 'LA']

    # Select 5 bonafide and 5 spoof (synthetic voice) files
    bonafides = la_df[la_df['label'] == 'bonafide'].head(5)
    spoofs = la_df[la_df['label'] == 'spoof'].head(5)

    test_files = pd.concat([bonafides, spoofs])

    detector = AIVoiceDetector()

    results = []
    print("=" * 80)
    print("EVALUATING DETECTOR_AI_VOICE (AASIST) ON 10 AUDIO FILES")
    print("=" * 80)

    correct_count = 0
    total_count = 0

    for idx, row in test_files.iterrows():
        fp = REPO_ROOT / row['subset_filepath']
        true_label = row['label']
        
        if not fp.exists():
            continue
            
        try:
            data, sr = sf.read(str(fp))
            res = detector.predict(data)
            score = res['score']
            top_feature = res['top_feature']
            
            pred_label = "spoof" if score >= 0.50 else "bonafide"
            is_correct = (score >= 0.50 and true_label == "spoof") or (score < 0.50 and true_label == "bonafide")
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            status_str = "CORRECT" if is_correct else "MISCLASSIFIED"
            
            results.append({
                "filename": fp.name,
                "true_label": true_label,
                "score": round(score, 4),
                "pred_label": pred_label,
                "status": status_str,
                "top_feature": top_feature
            })
            
            print(f"\n[{total_count}/10] File: {fp.name}")
            print(f"  True Label: {true_label.upper()}")
            print(f"  AI Voice / Synthetic Probability Score: {score:.4f} -> Pred Label: {pred_label.upper()}")
            print(f"  Result Status: [{status_str}]")
            print(f"  Explanation Driver: {top_feature}")
            
        except Exception as e:
            print(f"Error on {fp.name}: {e}")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {correct_count}/{total_count} files correctly classified ({correct_count/total_count * 100:.1f}% accuracy)")
    print("=" * 80)

if __name__ == "__main__":
    main()
