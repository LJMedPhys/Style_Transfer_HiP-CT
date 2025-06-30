import yaml
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src import hip_ct, clinical_ct

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)


    if config["hip_ct"].get("enable", True):
        hip_ct_config = config["hip_ct"]
        hip_ct.run(
            input_dir=hip_ct_config["input_dir"],
            mask_dir=hip_ct_config["mask_dir"],
            output_dir=hip_ct_config["output_dir"],
            output_mask_dir=hip_ct_config["output_mask_dir"],
            spacing=tuple(hip_ct_config["spacing"]),
            window_size=hip_ct_config["window_size"],
            stride=hip_ct_config["stride"],
            threshold=hip_ct_config["threshold"],
            percentile=hip_ct_config["percentile"],
            clip_values=tuple(hip_ct_config["clip_values"]),
            steps=hip_ct_config.get("steps", {}),
        )


    if config["clinical_ct"].get("enable", True):
        clinical_ct_config = config["clinical_ct"]
        clinical_ct.run(
            input_dir=clinical_ct_config["input_dir"],
            mask_dir=clinical_ct_config["mask_dir"],
            output_dir=clinical_ct_config["output_dir"],
            output_mask_dir=clinical_ct_config["output_mask_dir"],
            spacing=tuple(clinical_ct_config["spacing"]),
            window_size=clinical_ct_config["window_size"],
            stride=clinical_ct_config["stride"],
            padding=clinical_ct_config["padding"],
            threshold=clinical_ct_config["threshold"],
            percentile=clinical_ct_config["percentile"],
            cut_off=clinical_ct_config["cut_off"],
            steps=clinical_ct_config.get("steps", {}),
        )

if __name__ == "__main__":
    main()