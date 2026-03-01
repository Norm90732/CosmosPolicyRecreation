from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf
import hydra


"""
Downloads RoboCasa Dataset to downloadDir from config.dataset
"""


def downloadRoboCasa(cfg: DictConfig) -> None:
    snapshot_download(
        repo_id="nvidia/RoboCasa-Cosmos-Policy",
        repo_type="dataset",
        local_dir=cfg.dataset.downloadDir,
        max_workers=4,
    )
    return None


def main() -> None:
    cfg = OmegaConf.load("configs/config.yaml")  # pyrefly:ignore
    downloadRoboCasa(cfg)  # pyrefly:ignore

    return None


if __name__ == "__main__":
    main()
