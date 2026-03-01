from huggingface_hub import snapshot_download
from omegaconf import DictConfig
import hydra 

#download the weights to Checkpoints
def downloadCosmos2BWeights(cfg: DictConfig) -> None:
    snapshot_download(
        repo_id="nvidia/Cosmos-Predict2.5-2B",
        local_dir=cfg.model.baseModelWeightDownloadDir,
        allow_patterns=[
            "base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16*", 
            "*.json", 
            "*.md"
        ]
    )
    return None










@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:

    downloadCosmos2BWeights(cfg)

    return None


if __name__ == "__main__":
    main()