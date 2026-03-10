import pickle
import torch
from omegaconf import OmegaConf, DictConfig
import os
import glob
import numpy as np
from torch import Tensor
import webdataset as wds
from torch.utils.data import default_collate
from pathlib import Path
from functools import partial

"""
Reworked because of pickling issues when scaling. 
"""

_workerEmbValues = None
_workerEmbIndex = None

def _ensureMmapLoaded(mmapDir):
    global _workerEmbValues, _workerEmbIndex
    if _workerEmbValues is None:
        _workerEmbValues = np.load(
            Path(mmapDir, "embeddingvalues.npy"), mmap_mode="r"
        )
        embKeys = np.load(Path(mmapDir, "embeddingkeys.npy"), allow_pickle=True)
        _workerEmbIndex = {k: i for i, k in enumerate(embKeys)}


def _tagDemo(s):
    s["isdemo"] = b"1"
    return s

def _tagRollout(s):
    s["isdemo"] = b"0"
    return s

def _preprocessSample(sample, mmapDir):
    _ensureMmapLoaded(mmapDir)
    processedSample = {}
    processedSample["__key__"] = sample["__key__"]

    processedSample["futurevalue"] = torch.from_numpy(
        sample["futurevalue.npy"]
    ).float()
    processedSample["collectedactions"] = torch.from_numpy(
        sample["collectedactions.npy"]
    )
    processedSample["currentproprio"] = torch.from_numpy(
        sample["currentproprio.npy"]
    )
    processedSample["futureproprio"] = torch.from_numpy(
        sample["futureproprio.npy"]
    )
    processedSample["issuccess"] = torch.tensor(
        sample["issuccess.json"], dtype=torch.float32
    )

    textString = sample["taskdescription.txt"]
    idx = _workerEmbIndex[textString] #pyrefly:ignore
    lookedUpEmbedding = torch.from_numpy(_workerEmbValues[idx].copy()) #pyrefly:ignore
    processedSample["crossattentionembed"] = lookedUpEmbedding.squeeze(0)

    isDemoBool = sample["isdemo"] == b"1"
    processedSample["isdemo"] = torch.tensor(isDemoBool, dtype=torch.bool)

    imageKeys = [
        "currentwristimg.jpg",
        "currentleftimg.jpg",
        "currentrightimg.jpg",
        "futurewristimg.jpg",
        "futureleftimg.jpg",
        "futurerightimg.jpg",
    ]
    for key in imageKeys:
        image = sample[key]
        torchImage = torch.from_numpy(image.copy())
        torchImage = torchImage.permute(2, 0, 1)
        processedSample[key] = torchImage

    return processedSample


class RoboCasaWebDataset:
    def __init__(self, cfg: DictConfig):
        cfgResourcesDataloader = cfg.model.resources.dataloader
        self.batchSize = cfgResourcesDataloader.batchSize
        self.numWorkers = cfgResourcesDataloader.numWorkers
        self.prefetchFactor = cfgResourcesDataloader.prefetchFactor
        self.pinMemory = cfgResourcesDataloader.pinMemory

        self.successTars = sorted(
            glob.glob(os.path.join(cfg.dataset.allSuccessOutputDir, "*.tar"))
        )
        self.allScenesTars = sorted(
            glob.glob(os.path.join(cfg.dataset.allScenesOutputDir, "*.tar"))
        )

        self.mmapDir = str(cfg.dataset.reasonMMAP)
        self.onlySuccessSample = cfg.model.stage.onlySuccessSample
        self.allScenesSample = cfg.model.stage.allScenesSample

    def getDataloader(self):
        pipeDemo = wds.DataPipeline(
            wds.ResampledShards(self.successTars),
            wds.tarfile_to_samples(),
            wds.map(_tagDemo),
        )

        pipeRollout = wds.DataPipeline(
            wds.ResampledShards(self.allScenesTars),
            wds.tarfile_to_samples(),
            wds.map(_tagRollout),
        )

        mix = wds.RandomMix(
            [pipeDemo, pipeRollout],
            [self.onlySuccessSample, self.allScenesSample],
        )
        
        preprocessFn = partial(_preprocessSample, mmapDir=self.mmapDir)

        pipeline = wds.DataPipeline(
            mix,
            wds.decode("rgb8"),
            wds.map(preprocessFn),
            wds.batched(
                self.batchSize, collation_fn=default_collate, partial=False
            ),
        )

        return wds.WebLoader(
            pipeline,
            batch_size=None,
            num_workers=self.numWorkers,
            pin_memory=self.pinMemory,
            prefetch_factor=self.prefetchFactor if self.numWorkers > 0 else None,
            persistent_workers=True if self.numWorkers > 0 else False,
            multiprocessing_context="spawn",
        )


def mmapMaker(cfg: DictConfig) -> None:
    embeddingPath = cfg.dataset.reasonEmbeddings
    with open(embeddingPath, "rb") as f:
        d = pickle.load(f)
    keys = list(d.keys())
    embeddings = np.stack([
        d[k].to(torch.float32).numpy() if hasattr(d[k], 'numpy')
        else np.array(d[k], dtype=np.float32)
        for k in keys
    ]).astype(np.float32)
    saveDim = cfg.dataset.reasonMMAP
    np.save(Path(saveDim, "embeddingkeys.npy"), np.array(keys))
    np.save(Path(saveDim, "embeddingvalues.npy"), embeddings)
    
    
    return None 