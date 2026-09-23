import sys
import os
import json
from pathlib import Path
import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.detector_replay.infer import get_replay_detector

def main():
    metadata_path = REPO_ROOT / "data" / "subset_metadata.csv"
    df = pd.read_csv(metadata_path)

    # Filter for PA (Physical Access / Replay dataset)
    pa_bonafide_candidates = df[(df['dataset'] == 'PA') & (df['label'] == 'bonafide')]
    pa_spoof_candidates = df[(df['dataset'] == 'PA') & (df['label'] == 'spoof')]
    
    detector = get_replay_detector()
    
    print("=" * 80)
    print("RUNNING REPLAY DETECTOR ON 10 AUDIO FILES (5 Valid / Bonafide, 5 Replayed / Spoof)")
    print("=" * 80)

    selected_results = []
    
    def collect_samples(candidates, target_count, target_label):
        collected = []
        for idx, row in candidates.iterrows():
            if len(collected) >= target_count:
                break
            filepath = REPO_ROOT / row['subset_filepath']
            try:
                res = detector.predict_file(str(filepath))
                collected.append((row, res))
            except Exception as e:
                # Skip corrupted file
                print(f"  [SKIP CORRUPTED FILE] {filepath.name}: {e}")
                continue
        return collected

    print("\nProcessing 5 Valid (Bonafide) files...")
    bonafide_samples = collect_samples(pa_bonafide_candidates, 5, "bonafide")
    
    print("\nProcessing 5 Replayed (Spoof) files...")
    spoof_samples = collect_samples(pa_spoof_candidates, 5, "spoof")
    
    all_samples = bonafide_samples + spoof_samples
    
    correct_count = 0
    total_count = 0
    
    for row, res in all_samples:
        filepath = REPO_ROOT / row['subset_filepath']
        dataset = row['dataset']
        true_label = row['label'] # 'bonafide' or 'spoof'
        
        score = res['score']
        top_feature = res['top_feature']
        chunk_count = res['chunk_count']
        latency_ms = res['latency_ms']
        
        pred_label = "spoof (replay)" if score >= 0.50 else "bonafide (valid)"
        is_correct = (score >= 0.50 and true_label == "spoof") or (score < 0.50 and true_label == "bonafide")
        
        if is_correct:
            correct_count += 1
        total_count += 1
        
        status_str = "CORRECT" if is_correct else "MISCLASSIFIED"
        
        selected_results.append({
            "filename": filepath.name,
            "dataset": dataset,
            "true_label": true_label,
            "pred_score": score,
            "pred_label": pred_label,
            "status": status_str,
            "top_feature": top_feature,
            "latency_ms": latency_ms
        })
        
        print(f"\n[{total_count}/10] File: {filepath.name}")
        print(f"  Dataset: {dataset} | True Label: {true_label.upper()}")
        print(f"  Replay Probability Score: {score:.4f} -> Pred Label: {pred_label.upper()}")
        print(f"  Result Status: [{status_str}]")
        print(f"  Top Feature Driver: {top_feature}")
        print(f"  Latency: {latency_ms:.2f} ms ({chunk_count} chunks)")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {correct_count}/{total_count} files correctly classified ({correct_count/total_count * 100:.1f}% accuracy)")
    print("=" * 80)

    # Save detailed JSON for report reference
    output_report = REPO_ROOT / "scratch" / "replay_10_files_test_results.json"
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(selected_results, f, indent=2)

if __name__ == "__main__":
    main()
