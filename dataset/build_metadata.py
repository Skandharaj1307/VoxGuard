from pathlib import Path
import csv


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_ROOT = Path(r"E:\404NotFound\VoxGuard\data\raw")
OUTPUT_FILE = Path("data/metadata.csv")


# --------------------------------------------------
# Build audio-file index
# --------------------------------------------------

def build_audio_index(dataset_folder: Path):
    """
    Scan all audio files once and create a filename-to-path dictionary.
    """

    print(f"Indexing audio files inside: {dataset_folder}")

    audio_index = {}

    for path in dataset_folder.rglob("*.flac"):
        audio_index[path.stem] = path

    print(f"Indexed {len(audio_index)} audio files")

    return audio_index


# --------------------------------------------------
# Find protocol file
# --------------------------------------------------

def find_protocol_file(dataset_folder: Path, keyword: str) -> Path:
    """
    Search recursively for the protocol file.
    """

    keyword = keyword.lower()
    candidates = []

    for path in dataset_folder.rglob("*"):
        if not path.is_file():
            continue

        if keyword in path.name.lower():
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find a protocol file containing "
            f"'{keyword}' inside {dataset_folder}"
        )

    protocol_candidates = [
        path for path in candidates
        if "protocol" in str(path).lower()
    ]

    selected_file = (
        protocol_candidates[0]
        if protocol_candidates
        else candidates[0]
    )

    print(f"Using protocol file: {selected_file}")

    return selected_file


# --------------------------------------------------
# Process protocol
# --------------------------------------------------

def process_protocol(
    dataset_folder: Path,
    protocol_keyword: str,
    dataset_name: str,
    metadata_rows: list,
):
    """
    Process a protocol using a prebuilt audio index.
    """

    if not dataset_folder.exists():
        print(f"WARNING: Folder does not exist: {dataset_folder}")
        return

    # Build index once
    audio_index = build_audio_index(dataset_folder)

    protocol_file = find_protocol_file(
        dataset_folder,
        protocol_keyword,
    )

    print(f"Reading protocol: {protocol_file}")

    processed_count = 0
    missing_audio_count = 0

    with protocol_file.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            columns = line.split()

            if len(columns) < 5:
                print(f"Skipping malformed line {line_number}")
                continue

            speaker_id = columns[0]
            audio_filename = columns[1]
            attack_id = columns[-2]
            label = columns[-1].lower()

            # Remove extension if present
            audio_key = Path(audio_filename).stem

            audio_path = audio_index.get(audio_key)

            if audio_path is None:
                missing_audio_count += 1
                continue

            attack_category = (
                "bonafide"
                if label == "bonafide"
                else "spoof"
            )

            metadata_rows.append(
                {
                    "filepath": str(audio_path.resolve()),
                    "dataset": dataset_name,
                    "speaker_id": speaker_id,
                    "attack_id": attack_id,
                    "attack_category": attack_category,
                    "label": label,
                }
            )

            processed_count += 1

            if processed_count % 1000 == 0:
                print(
                    f"{dataset_name}: processed "
                    f"{processed_count} files..."
                )

    print(f"{dataset_name} processed: {processed_count}")
    print(f"{dataset_name} missing audio: {missing_audio_count}")


# --------------------------------------------------
# Save metadata
# --------------------------------------------------

def save_metadata(rows: list, output_file: Path):
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "filepath",
        "dataset",
        "speaker_id",
        "attack_id",
        "attack_category",
        "label",
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMetadata saved to: {output_file.resolve()}")
    print(f"Total rows: {len(rows)}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    metadata_rows = []

    la_folder = DATASET_ROOT / "LA" / "LA"
    pa_folder = DATASET_ROOT / "PA" / "PA"

    print(f"LA folder: {la_folder}")
    print(f"PA folder: {pa_folder}")

    process_protocol(
        dataset_folder=la_folder,
        protocol_keyword="LA.cm.train.trn",
        dataset_name="LA",
        metadata_rows=metadata_rows,
    )

    process_protocol(
        dataset_folder=pa_folder,
        protocol_keyword="PA.cm.train.trn",
        dataset_name="PA",
        metadata_rows=metadata_rows,
    )

    save_metadata(
        rows=metadata_rows,
        output_file=OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()