from huggingface_hub import snapshot_download
from omegaconf import DictConfig
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:

    downloadRoboCasa(cfg)

    return None


if __name__ == "__main__":
    main()
