import torch
from omegaconf import DictConfig,OmegaConf

#distributed training 


#logging


#Constructed Modules Imports 
from engine.train.builder import LatentSequenceBuilder
from engine.train.diffusion import EDMNoiseScheduler
from engine.train.dataloader import RoboCasaWebDataset
from engine.train.trainerUtils import buildConditioningMasks,optimizerAndSchedulerCreator
from models.cosmosLoader import EncoderVAE,CosmosDiffusionNet,ReasonTextEncoder





def trainFunction(cfg:DictConfig):
    pass 