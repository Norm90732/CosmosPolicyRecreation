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
        self.successTars = sorted(glob.glob(
            os.path.join(cfg.dataset.allSuccessOutputDir, "*.tar")
        ))
        self.allScenesTars = sorted(glob.glob(
            os.path.join(cfg.dataset.allScenesOutputDir, "*.tar")
        ))

        with open(self.embeddingPath, "rb") as f:
            self.textEmbeddingsDict = pickle.load(f)

        #Conditioning masks 
        
        
        self.stage = cfg.model.stage.stageNumber
        
        
        self.onlySuccessSample = cfg.model.stage.onlySuccessSample
        self.allScenesSample = cfg.model.stage.allScenesSample
                
        
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
        processedSample["isdemo"] = torch.tensor(sample["isdemo"], dtype=torch.bool)
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
        def tagDemo(s): s['isdemo'] = True; return s
        def tagrollout(s): s['isdemo'] = False; return s
        
        pipeDemo = wds.DataPipeline(
             wds.ResampledShards(self.successTars),
            wds.tarfile_to_samples(),
            wds.map(tagDemo)
        )
        
        pipeRollout = wds.DataPipeline(
             wds.ResampledShards(self.allScenesTars),
            wds.tarfile_to_samples(),
            wds.map(tagrollout)
        )
        
        mix = wds.RandomMix([
            pipeDemo,pipeRollout],[self.onlySuccessSample,self.allScenesSample]
        )
        
        pipeline = wds.DataPipeline(
            mix,
            wds.decode("rgb8"),
            wds.map(self._preprocessSample), 
            wds.batched(self.batchSize, collation_fn=default_collate, partial=False),
        )
        
        return wds.WebLoader(pipeline, batch_size=None, num_workers=self.numWorkers, pin_memory=self.pinMemory)
        
        