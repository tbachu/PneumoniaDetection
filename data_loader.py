from pathlib import Path
import shutil

import kagglehub


# Download the latest version to kagglehub cache, then move it into this project.
source_path = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")).resolve()
project_root = Path(__file__).resolve().parent
target_path = project_root / "data" / "chest-xray-pneumonia"

target_path.parent.mkdir(parents=True, exist_ok=True)

if target_path.exists():
	shutil.rmtree(target_path)

if source_path != target_path:
	shutil.move(str(source_path), str(target_path))

print("Dataset is now at:", target_path)