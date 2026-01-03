import os, zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

token = os.environ["HF_TOKEN"]
target_dir = Path(r"gpt_sovits_models/G2PWModel")
target_dir.mkdir(parents=True, exist_ok=True)

zip_path = hf_hub_download(
    repo_id="XXXXRT/GPT-SoVITS-Pretrained",
    filename="G2PWModel.zip",
    token=token,
    local_dir=str(target_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
)
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(target_dir)

print("G2PWModel ready at", target_dir.resolve())
