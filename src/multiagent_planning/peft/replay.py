import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

logger = logging.getLogger(__name__)

def replay_and_validate_trajectory(trajectory_file_path: str) -> bool:
    """
    Trajectory Replay: Reads exported JSONL files, validates structures,
    and runs a formatting pass to verify downstream training compatibility.
    """
    path = Path(trajectory_file_path)
    if not path.exists():
        logger.error(f"Trajectory file not found: {trajectory_file_path}")
        return False

    logger.info(f"Replaying trajectory dataset: {trajectory_file_path}")
    
    records = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Parsing trajectory file"):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode failure inside trajectory record: {e}")
                    return False

    logger.info(f"Loaded {len(records)} records. Checking formatting...")
    
    for idx, record in enumerate(records):
        # 1. Structure Verification
        if "prompt" not in record or "completion" not in record:
            logger.error(f"Format mismatch at record index {idx}. Missing 'prompt' or 'completion'.")
            return False
            
        # Log embedded profile information if present
        if "profile_metrics" in record:
            prof = record["profile_metrics"]
            logger.info(f"Record {idx} Profile: Duration={prof.get('duration_sec'):.4f}s, Peak VRAM={prof.get('peak_vram_allocated_mb', 0):.2f}MB")
    
    # 2. Dataset Compilation Test
    try:
        dataset = Dataset.from_list(records)
        
        def format_prompts(example):
            return {"text": f"Prompt: {example['prompt']}\nCompletion: {example['completion']}"}
            
        formatted = dataset.map(format_prompts, desc="Replaying instruction SFT formatting map")
        logger.info(f"Successfully compiled {len(formatted)} formatting training records.")
    except Exception as e:
        logger.error(f"Failed compiling HuggingFace SFT Dataset format: {e}")
        return False

    logger.info("Trajectory SFT compliance validation: SUCCESS.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and replay trajectory logs for SFT/DPO alignment reproducibility.")
    parser.add_argument("--file", type=str, required=True, help="Path to exported JSONL trajectory log")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    success = replay_and_validate_trajectory(args.file)
    if not success:
        logger.error("Trajectory verification failed.")
        exit(1)
