from pathlib import Path

import wandb
from tqdm import tqdm


api = wandb.Api()
run = api.run()

src_dir = Path("/root/workspace")
[
    run.upload_file(file)
    for file in tqdm(list(src_dir.rglob("*")), desc="Uploading files")
    if not file.is_dir() and file.suffix in [".py", ".json", ".yaml"]
]
