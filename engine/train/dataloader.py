import pickle
import torch
from omegaconf import OmegaConf, DictConfig
import os
import glob
import numpy as np
from torch import Tensor
import webdataset as wds
from torch.utils.data import default_collate


class RoboCasaWebDataset:
    def __init__(self, cfg: DictConfig):
        self.embeddingPath = cfg.dataset.reasonEmbeddings
        self.datasetName = cfg.model.datasetName

        # batchSize, numworkers, prefetchFactor, pinmemory from config
        cfgResourcesDataloader = cfg.model.resources.dataloader

        self.batchSize = cfgResourcesDataloader.batchSize
        self.numWorkers = cfgResourcesDataloader.numWorkers
        self.prefetchFactor = cfgResourcesDataloader.prefetchFactor
        self.pinMemory = cfgResourcesDataloader.pinMemory

        # Dataset loading
        if self.datasetName == "success_only":
            chosenDirectories = [cfg.dataset.allSuccessOutputDir]

        elif self.datasetName == "all_scenes":
            chosenDirectories = [cfg.dataset.allScenesOutputDir]

        elif self.datasetName == "both":
            chosenDirectories = [
                cfg.dataset.allSuccessOutputDir,
                cfg.dataset.allScenesOutputDir,
            ]
        else:
            raise ValueError(f"Unknown datasetName: {self.datasetName}")
        allTarFiles = []

        for currentDir in chosenDirectories:
            if not os.path.exists(currentDir):
                continue
            tars = sorted(glob.glob(os.path.join(currentDir, "*.tar")))
            allTarFiles.extend(tars)

        self.allTarFiles = allTarFiles

        with open(self.embeddingPath, "rb") as f:
            self.textEmbeddingsDict = pickle.load(f)

    def _preprocessSample(self, sample):
        # define new Dictionary to return
        processedSample = {}
        # line up key
        processedSample["__key__"] = sample["__key__"]

        # process actions and proprio
        processedSample["futurevalue"] = torch.from_numpy(sample["futurevalue.npy"]).float()
        processedSample["collectedactions"] = torch.from_numpy(
            sample["collectedactions.npy"]
        )
        processedSample["currentproprio"] = torch.from_numpy(
            sample["currentproprio.npy"]
        )
        processedSample["futureproprio"] = torch.from_numpy(sample["futureproprio.npy"])
        processedSample["issuccess"] = torch.tensor(
            sample["issuccess.json"], dtype=torch.float32
        )
        # fetch textEmbedding
        textString = sample["taskdescription.txt"]

        lookedUpEmbedding = self.textEmbeddingsDict[textString]

        processedSample["crossattentionembed"] = lookedUpEmbedding.squeeze(0)

        # img processing
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

    def getDataloader(self):
        # .decode("rgb8")
        pipeline = [
            wds.ResampledShards(self.allTarFiles),
            wds.shuffle(100),
            wds.tarfile_to_samples(),
            wds.decode("rgb8"),
            wds.map(self._preprocessSample),
            wds.batched(self.batchSize, collation_fn=default_collate, partial=False),
        ]

        dataset = wds.DataPipeline(*pipeline)

        dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            num_workers=self.numWorkers,
            pin_memory=self.pinMemory,
            prefetch_factor=self.prefetchFactor,
        )

        return dataloader


