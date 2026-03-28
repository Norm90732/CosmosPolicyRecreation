from multiprocessing import Value
import torch
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    SequentialLR,
    LinearLR,
    ConstantLR,
)
from torch import Tensor
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
import numpy as np

"""
conditioningMasks = {
            "policy":         [0,1, 2, 3, 4], # s
            "worldModel":    [0,1, 2, 3, 4, 5], # s + a
            "valueFunction": [0,1, 2, 3, 4, 5, 6, 7, 8, 9], # s + a + s'
        }
"""


def buildConditioningMasks(
    isDemo: torch.Tensor,
    device: torch.device,
    stage: int,
    valueFunctionVariant: str = "base",
    T: int = 11,
    H: int = 28,
    W: int = 28,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    valueFunctionVariant:
        base: value conditioned on (s, a, s')
        Vs: value conditioned only on s'
        Qsa: value conditioned only on (s, a)
    """
    B = isDemo.shape[0]

    conditionVideoMask = torch.zeros(
        (B, 1, T, H, W), device=device, dtype=torch.bfloat16
    )

    conditionVideoMask[:, :, :5, :, :] = (
        1.0  # base that applies to policy, world model, and value function
    )

    randomValues = torch.rand(B, device=device)

    # paper uses a 50 | 25, 25 split for stage 1
    if stage == 1:
        isPolicy = isDemo

        isWorldModel = (~isDemo) & (randomValues < 0.5)  # mask

        isValueFunction = (~isDemo) & (randomValues >= 0.5)

    # paper uses a 10 | 45, 45 split for stage 2
    elif stage == 2:
        # isDemo = false
        isPolicy = randomValues < 0.10
        isWorldModel = (randomValues >= 0.10) & (randomValues < 0.55)
        isValueFunction = randomValues >= 0.55

    conditionVideoMask[isWorldModel, :, 5, :, :] = 1.0  # pyrefly:ignore

    if valueFunctionVariant == "base":
        # condition on full sequence to predict the value
        conditionVideoMask[isValueFunction, :, 5:10, :, :] = 1.0  # pyrefly:ignore

    elif valueFunctionVariant == "Vs":
        # condition on s'
        conditionVideoMask[isValueFunction, :, :6, :, :] = 0.0  # pyrefly:ignore
        conditionVideoMask[isValueFunction, :, 6:10, :, :] = 1.0  # pyrefly:ignore

    elif valueFunctionVariant == "Qsa":
        # condition on s,a, predict s'
        conditionVideoMask[isValueFunction, :, :6, :, :] = 1.0  # pyrefly:ignore
        conditionVideoMask[isValueFunction, :, 6:10, :, :] = 0.0  # pyrefly:ignore

    return conditionVideoMask, isPolicy, isWorldModel, isValueFunction  # pyrefly:ignore


""""
Training Loop for RecFlow or EDM to abstract away from trainer
"""


class NoiseTrainer:
    def __init__(self, trainingStyle: str, noiseScheduler):
        if trainingStyle not in ("edm", "recflow"):
            raise ValueError("incorrectTraining Style selected")

        self.trainingStyle = trainingStyle
        self.noiseScheduler = noiseScheduler

    def _EdmForward(
        self,
        model,
        vaeOutput,
        crossAttentionEmbed,
        device,
        conditioningMasks,
        paddingMask,
        batchSize,
        T,
    ):
        sigma = self.noiseScheduler.sampleSigma(batchSize, device=device)
        skipSigma, outputSigma, cinSigma, noiseSigma, lossWeighting = (
            self.noiseScheduler.EDMScalingFactors(sigma)
        )

        sigma = sigma.view(batchSize, 1, 1, 1, 1)
        skipSigma = skipSigma.view(batchSize, 1, 1, 1, 1)
        outputSigma = outputSigma.view(batchSize, 1, 1, 1, 1)
        cinSigma = cinSigma.view(batchSize, 1, 1, 1, 1)
        noiseSigma = noiseSigma.view(batchSize, 1).repeat(1, T)  # b,T
        lossWeighting = lossWeighting.view(batchSize, 1, 1, 1, 1)
        epsilon = torch.randn_like(vaeOutput)

        xNoise = vaeOutput + (sigma * epsilon)  # xSigma = y + sigma * eps

        # masking the input
        maskedXSigma = conditioningMasks * vaeOutput + (1 - conditioningMasks) * xNoise
        networkInput = maskedXSigma * cinSigma

        networkOutput = model.forward(
            latent=networkInput,
            timesteps=noiseSigma,
            crossAttentionEmbed=crossAttentionEmbed,
            conditionVideoMask=conditioningMasks,
            paddingMask=paddingMask,
        )

        denoisedPrediction = (maskedXSigma * skipSigma) + (networkOutput * outputSigma)

        loss = lossFunctionWeighting(
            lossWeighting, vaeOutput, denoisedPrediction, conditioningMasks
        )

        return loss, denoisedPrediction

    def _RecFlowForward(
        self,
        model,
        vaeOutput,
        crossAttentionEmbed,
        device,
        conditioningMasks,
        paddingMask,
        batchSize,
        T,
    ):
        t = self.noiseScheduler.sampleTimestep(batchSize, device=device)
        tExpand = t.view(batchSize, 1, 1, 1, 1)
        tInput = t.view(batchSize, 1).repeat(1, T)
        x0Noise = torch.randn_like(vaeOutput)

        x0 = conditioningMasks * vaeOutput + (1 - conditioningMasks) * x0Noise
        x1 = vaeOutput

        target = x0 - x1

        xt = (1 - tExpand) * x1 + (tExpand * x0)

        velocityPred = model.forward(
            latent=xt,
            timesteps=tInput,
            crossAttentionEmbed=crossAttentionEmbed,
            conditionVideoMask=conditioningMasks,
            paddingMask=paddingMask,
        )

        loss = lossFunctionUnWeighting(target, velocityPred, conditioningMasks)
        with torch.no_grad():
            x1Prediction = xt - tExpand * velocityPred

        return loss, x1Prediction

    def forward(
        self,
        model,
        vaeOutput,
        crossAttentionEmbed,
        device,
        conditioningMasks,
        paddingMask,
        batchSize,
        T,
    ):
        if self.trainingStyle == "edm":
            return self._EdmForward(
                model,
                vaeOutput,
                crossAttentionEmbed,
                device,
                conditioningMasks,
                paddingMask,
                batchSize,
                T,
            )
        else:
            return self._RecFlowForward(
                model,
                vaeOutput,
                crossAttentionEmbed,
                device,
                conditioningMasks,
                paddingMask,
                batchSize,
                T,
            )


def optimizerAndSchedulerCreator(model, cfg: DictConfig) -> tuple[AdamW, SequentialLR]:
    cfgOptimizer = cfg.model.training.optimizer
    cfgScheduler = cfg.model.training.scheduler

    optimizer = AdamW(
        model.parameters(),
        lr=cfgOptimizer.lr,
        betas=tuple(cfgOptimizer.betas),  # pyrefly:ignore
        eps=cfgOptimizer.eps,
        weight_decay=cfgOptimizer.weight_decay,
    )

    warmUpScheduler = LinearLR(
        optimizer,
        start_factor=cfgScheduler.linear.startFactor,
        end_factor=cfgScheduler.linear.endFactor,
        total_iters=cfgScheduler.linear.totalIters,
    )

    cosineScheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfgScheduler.cosine.TMax,
        eta_min=cfgScheduler.cosine.etaMin * cfgOptimizer.lr,
    )

    flatlineScheduler = ConstantLR(
        optimizer,
        factor=cfgScheduler.flatlineScheduler.factor,
        total_iters=10**9,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmUpScheduler, cosineScheduler, flatlineScheduler],
        milestones=[
            cfgScheduler.linear.totalIters,
            cfgScheduler.linear.totalIters + cfgScheduler.cosine.TMax,
        ],
    )

    return optimizer, scheduler


"""
No EMA on the policy training 
"""


def unwrapModel(model):
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def l1LossGeneration(
    vaeOutput,
    denoisedPrediction,
    conditioningMasks,
    isPolicy,
    isWorldModel,
    isValueFunction,
):
    B, C, T, H, W = vaeOutput.shape
    targetMask = 1.0 - conditioningMasks
    absMAE = (denoisedPrediction - vaeOutput).abs()

    def maskedL1(frameIdxs, sampleMask=None):
        frameMask = torch.zeros_like(targetMask)
        frameMask[:, :, frameIdxs, :, :] = 1.0
        objMask = frameMask * targetMask

        if sampleMask is not None:
            objMask = objMask * sampleMask[:, None, None, None, None].float()

        objError = absMAE * objMask
        denom = (objMask.sum(dim=[1, 2, 3, 4]) * C).clamp_min(1.0)
        perSampleL1 = objError.sum(dim=[1, 2, 3, 4]) / denom
        return perSampleL1.mean()

    metrics = {
        "l1ActionPolicy": maskedL1([5], isPolicy),
        "l1FutureProprioPolicy": maskedL1([6], isPolicy),
        "l1WristLatentPolicy": maskedL1([7], isPolicy),
        "l1ThirdLatentsPolicy": maskedL1([8, 9], isPolicy),
        "l1ValuePolicy": maskedL1([10], isPolicy),
        "l1FutureProprioWM": maskedL1([6], isWorldModel),
        "l1WristLatentWM": maskedL1([7], isWorldModel),
        "l1ThirdLatentsWM": maskedL1([8, 9], isWorldModel),
        "l1ValueWM": maskedL1([10], isWorldModel),
        "l1ValueVF": maskedL1([10], isValueFunction),
    }

    metrics["l1FutureProprio"] = (
        metrics["l1FutureProprioPolicy"] + metrics["l1FutureProprioWM"]
    ) / 2
    metrics["l1WristLatent"] = (
        metrics["l1WristLatentPolicy"] + metrics["l1WristLatentWM"]
    ) / 2
    metrics["l1ThirdLatents"] = (
        metrics["l1ThirdLatentsPolicy"] + metrics["l1ThirdLatentsWM"]
    ) / 2
    metrics["l1Value"] = (
        metrics["l1ValuePolicy"] + metrics["l1ValueWM"] + metrics["l1ValueVF"]
    ) / 3

    return metrics


def lossFunctionWeighting(
    lossWeighting, vaeOutput, denoisedPrediction, conditioningMasks
):
    B, C, T, H, W = vaeOutput.shape
    targetMask = 1.0 - conditioningMasks
    perPixelMSE = (denoisedPrediction - vaeOutput) ** 2
    maskedMSE = perPixelMSE * targetMask * lossWeighting
    loss = maskedMSE.mean()

    return loss

# from source code. 
def lossFunctionUnWeighting(vaeOutput, denoisedPrediction, conditioningMasks,actionMultiplier:int=16):
    B, C, T, H, W = vaeOutput.shape
    targetMask = 1.0 - conditioningMasks
    perFrameWeight = torch.ones(B, T, device=vaeOutput.device, dtype=vaeOutput.dtype)
    perFrameWeight[:, 5] = actionMultiplier #config later 
    
    perPixelMSE = (denoisedPrediction - vaeOutput) ** 2
    maskedMSE = perPixelMSE * targetMask * perFrameWeight[:, None, :, None, None]
    loss = maskedMSE.mean()

    return loss


# default values that Cosmos Policy Uses, config this later.
def createAugmentationPipeline():
    strongAugmentationPipeline = v2.Compose(
        [
            v2.RandomResizedCrop(
                size=(224, 224), scale=(0.9, 0.9), ratio=(1.0, 1.0), antialias=True
            ),
            v2.RandomRotation(degrees=5),  # pyrefly:ignore
            v2.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.05),
        ]
    )
    return strongAugmentationPipeline


def applyUnifiedCameraAug(*images, pipeline: v2.Compose):
    stackedTensor = torch.stack(images, dim=1)
    videoWrapped = tv_tensors.Video(stackedTensor)
    augmentedVideo = pipeline(videoWrapped)
    return torch.unbind(augmentedVideo, dim=1)


# gpu collator
def gpuCollate(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def loadEmbeddingTableToGPU(cfg, device):
    from pathlib import Path

    embValues = np.load(Path(cfg.dataset.reasonMMAP, "embeddingvalues.npy"))
    table = torch.from_numpy(embValues).to(device=device, dtype=torch.bfloat16)
    if table.dim() == 4:
        table = table.squeeze(1)
    return table
