import os
os.environ["JAXTYPING_DISABLE"] = "1"
os.environ["RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S"] = "5.0"
os.environ["RAY_TRAIN_WORKER_HEALTH_CHECK_TIMEOUT_S"] = "3600"
os.environ["RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S"] = "3600"
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from einops import rearrange, repeat
import numpy as np
import gc
# distributed training
import ray
import ray.train
import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, Checkpoint,CheckpointConfig
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

# logging
import wandb
import tempfile
import io

# Constructed Modules Imports
from engine.train.builder import LatentSequenceBuilder
from engine.train.diffusion import EDMNoiseScheduler
from engine.train.dataloader import RoboCasaWebDataset
from engine.train.trainerUtils import (
    buildConditioningMasks,
    optimizerAndSchedulerCreator,
    unwrapModel,
    l1LossGeneration,
    lossFunctionWeighting,
    createAugmentationPipeline,
    applyUnifiedCameraAug,
    gpuCollate,
    loadEmbeddingTableToGPU
)
from models.cosmosLoaderBase import EncoderVAE, CosmosDiffusionNet, loadCosmosModules



def trainingFunction(config: dict):
    cfg = config["cfg"]
    rank = ray.train.get_context().get_world_rank()
    worldSize = ray.train.get_context().get_world_size()
    device = torch.device(f"cuda:{ray.train.get_context().get_local_rank()}")


    
    if rank == 0:
        wandb.init(
            project=cfg.model.logging.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),  # pyrefly:ignore
            name=cfg.model.logging.wandb.name,
            resume="allow"
        )

    print("starting model load on rank", rank)
    # constants def
    stage = cfg.model.stage.stageNumber
    valueFunctionVariant = str(cfg.model.stage.valueFunctionVariant)
    gradAccum = cfg.model.training.globalBatch // (worldSize * cfg.model.resources.dataloader.batchSize)
    gradClip = cfg.model.training.gradClip
    l1LogEvery = cfg.model.logging.l1LogEvery
    checkpointEvery = cfg.model.logging.checkpointing.checkpointEvery
    
    vae, textEncoder, net, config = loadCosmosModules(cfg)  # pyrefly:ignore

    # define VAE
    modelVAE = EncoderVAE(vae)

    # define modelDiT
    modelNet = CosmosDiffusionNet(net)

    # delete text encoder since embeddings are precomputed
    del textEncoder
    torch.cuda.empty_cache()

    gc.collect()

    # ddp wrap
    model = ray.train.torch.prepare_model(
        modelNet, parallel_strategy_kwargs={"find_unused_parameters": False}
    )

    # pass VAE to latent builder
    latentBuilder = LatentSequenceBuilder(cfg=cfg, vaeModel=modelVAE, device=device).to(
        device
    )

    # noise scheduler creation
    noiseScheduler = EDMNoiseScheduler(cfg=cfg, inference=False, device=device)

    optimizer, scheduler = optimizerAndSchedulerCreator(model, cfg)

    # dataloader creation
    robocasaDataloader = RoboCasaWebDataset(cfg=cfg)

    trainDataloader = robocasaDataloader.getDataloader()
    
    #augmentation maker 
    augmentationPipeline = createAugmentationPipeline()
    embeddingTableGPU = loadEmbeddingTableToGPU(cfg, device)
    # checkpoint resumption and start logic
    resumePath = cfg.model.logging.checkpointing.get("resumeCheckpoint", None)
    if resumePath and os.path.exists(resumePath):
        checkpointFile = os.path.join(resumePath, "checkpoint.pt")
        if os.path.exists(checkpointFile):
            ckpt = torch.load(checkpointFile, map_location=device, weights_only=False)
        else:
            ckpt = torch.load(resumePath, map_location=device, weights_only=False)
        unwrapModel(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        globalStep = ckpt["globalStep"]
    else:
        globalStep = 0

    model.train()
    optimizer.zero_grad()
    for step, batch in enumerate(trainDataloader):
        batch = gpuCollate(batch,device)
        # actions
        currentProprio = batch["currentproprio"].to(device, non_blocking=True)
        futureProprio = batch["futureproprio"].to(device, non_blocking=True)
        collectedActions = batch["collectedactions"].to(device, non_blocking=True)
        futureValue = batch["futurevalue"].to(device, non_blocking=True)
        # success and demo flags
        isSuccess = batch["issuccess"].to(
            device, non_blocking=True
        )  # dont need delete later.
        isDemo = batch["isdemo"].to(device, non_blocking=True)
        # cross attention embedding
        embIdx = batch["embeddingidx"].long()
        crossAttentionEmbed = embeddingTableGPU[embIdx]
        # current images
        currentWristImg = batch["currentwristimg.jpg"].to(device, non_blocking=True)
        currentLeftImg = batch["currentleftimg.jpg"].to(device, non_blocking=True)
        currentRightImg = batch["currentrightimg.jpg"].to(device, non_blocking=True)
        # future images
        futureWristImg = batch["futurewristimg.jpg"].to(device, non_blocking=True)
        futureLeftImg = batch["futureleftimg.jpg"].to(device, non_blocking=True)
        futureRightImg = batch["futurerightimg.jpg"].to(device, non_blocking=True)
        
        #apply gpu augmentations 
        (currentWristImg, currentLeftImg, currentRightImg, futureWristImg, 
         futureLeftImg, futureRightImg) = applyUnifiedCameraAug(
            currentWristImg, 
            currentLeftImg, 
            currentRightImg, 
            futureWristImg, 
            futureLeftImg, 
            futureRightImg,
            pipeline=augmentationPipeline
        )
        
        vaeOutput = latentBuilder(  # x0 = VAE Output
            currentProprio=currentProprio,
            currentWristImg=currentWristImg,
            currentLeftImg=currentLeftImg,
            currentRightImg=currentRightImg,
            collectedActions=collectedActions,
            futureProprio=futureProprio,
            futureWristImg=futureWristImg,
            futureLeftImg=futureLeftImg,
            futureRightImg=futureRightImg,
            futureValue=futureValue,
        )

        conditioningMasks, isPolicy, isWorldModel, isValueFunction = (
            buildConditioningMasks(
                isDemo=isDemo,
                device=device,
                stage=stage,
                valueFunctionVariant=valueFunctionVariant,
            )
        )

        # model noising and target creation
        batchSize = currentWristImg.shape[0]
        T = vaeOutput.shape[2]
        sigma = noiseScheduler.sampleSigma(batchSize, device=device)
        skipSigma, outputSigma, cinSigma, noiseSigma, lossWeighting = (
            noiseScheduler.EDMScalingFactors(sigma)
        )
        # b -> b,1,1,1,1

        # reshaping
        
        sigma = sigma.view(batchSize, 1, 1, 1, 1)
        skipSigma = skipSigma.view(batchSize, 1, 1, 1, 1)
        outputSigma = outputSigma.view(batchSize, 1, 1, 1, 1)
        cinSigma = cinSigma.view(batchSize, 1, 1, 1, 1)
        noiseSigma = noiseSigma.view(batchSize, 1).repeat(1, T) #b,T
        lossWeighting = lossWeighting.view(batchSize, 1, 1, 1, 1)
        epsilon = torch.randn_like(vaeOutput)

        xNoise = vaeOutput + (sigma * epsilon)  # xSigma = y + sigma * eps

        # masking the input
        maskedXSigma = conditioningMasks * vaeOutput + (1 - conditioningMasks) * xNoise
        networkInput = maskedXSigma * cinSigma

        
        #padding mask 
        B, C, T, H, W = vaeOutput.shape
        paddingMask = torch.zeros(
            B,1,H,W,device=device,dtype=torch.bfloat16,  
        )
        
        
        # modeloutput
        networkOutput = model.forward(
            latent=networkInput,
            timesteps=noiseSigma,
            crossAttentionEmbed=crossAttentionEmbed,
            conditionVideoMask=conditioningMasks,
            paddingMask=paddingMask
        )

        denoisedPrediction = (maskedXSigma * skipSigma) + (networkOutput * outputSigma)

        # mse loss with lambda scaling

        loss = lossFunctionWeighting(
            lossWeighting, vaeOutput, denoisedPrediction, conditioningMasks
        )
        
        scaledLoss = loss / gradAccum
        scaledLoss.backward()
        
        if (step + 1) % gradAccum == 0:
            clippedGradNorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradClip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            globalStep += 1
            
            
            reportMetrics = {
                "trainLoss": loss.item(),
                "globalStep": globalStep,
            }
            checkpoint = None
            if globalStep % checkpointEvery == 0:
                if rank == 0:
                    checkpointDict = {
                        "model": unwrapModel(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "globalStep": globalStep,
                    }
                    tmpdir = tempfile.mkdtemp()
                    torch.save(checkpointDict, os.path.join(tmpdir, f"checkpoint_{globalStep}.pt"))
                    checkpoint = ray.train.Checkpoint.from_directory(tmpdir)

            ray.train.report(reportMetrics, checkpoint=checkpoint)

            if rank == 0:
                logDict = {
                    "trainLoss": loss.item(),
                    "trainGradNorm": clippedGradNorm.item(),
                    "trainLr": scheduler.get_last_lr()[0],
                    "globalStep": globalStep,
                }

                if globalStep % l1LogEvery == 0:
                    l1Metrics = l1LossGeneration(
                        vaeOutput=vaeOutput,
                        denoisedPrediction=denoisedPrediction,
                        conditioningMasks=conditioningMasks,
                        isPolicy=isPolicy,
                        isWorldModel=isWorldModel,
                        isValueFunction=isValueFunction,
                    )
                    for k, v in l1Metrics.items():
                        logDict[f"train/l1/{k}"] = v.item()

                wandb.log(logDict)
                if globalStep >= cfg.model.training.maxSteps:
                    if rank == 0:
                        checkpointDict = {
                            "model": unwrapModel(model).state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "globalStep": globalStep,
                        }
                        torch.save(checkpointDict, os.path.join(cfg.model.logging.checkpointing.checkpointSaveDir, "final_47k_model.pt"))
                    break
        
                
    if rank == 0:
        wandb.finish()    
        
if __name__ == "__main__":
    cfg = OmegaConf.load("configs/config.yaml")
    cfg.dataset = OmegaConf.load("configs/dataset/robocasa.yaml")
    cfg.model = OmegaConf.load("configs/model/policy.yaml")
    cfg.inference = OmegaConf.load("configs/inference/rollout.yaml")
    #ray 
    
    ray.init( #pyrefly:ignore
        ignore_reinit_error=True #pyrefly:ignore
    )

    trainConfig = {
        "cfg": cfg,
    }
    
    trainer = TorchTrainer(
        train_loop_per_worker=trainingFunction,
        train_loop_config=trainConfig,
        scaling_config=ScalingConfig(  # pyrefly:ignore
            num_workers=cfg.model.resources.scaling.totalWorkers,
            use_gpu=True,
            resources_per_worker={
                "CPU": cfg.model.resources.scaling.resourcesPerWorker.CPU,
                "GPU": cfg.model.resources.scaling.resourcesPerWorker.GPU,
            },
        ),
        run_config=RunConfig(  # pyrefly:ignore
            name=cfg.model.logging.wandb.name,
            storage_path=cfg.model.logging.checkpointing.checkpointSaveDir,
            checkpoint_config=CheckpointConfig(  # pyrefly:ignore
                num_to_keep=cfg.model.logging.checkpointing.numToKeep,
                checkpoint_score_attribute="trainLoss",
                checkpoint_score_order="min",
            ),
        ),
    )

    result = trainer.fit()

    ray.shutdown()  # pyrefly:ignore
    