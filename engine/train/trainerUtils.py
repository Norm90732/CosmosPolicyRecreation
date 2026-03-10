import torch
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    SequentialLR,
    LinearLR,
    ConstantLR,
)
from torch import Tensor


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


def optimizerAndSchedulerCreator(model, cfg: DictConfig) -> tuple[Adam, SequentialLR]:
    cfgOptimizer = cfg.model.training.optimizer
    cfgScheduler = cfg.model.training.scheduler

    optimizer = Adam(
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
    lossPerSample = maskedMSE.sum(dim=[1, 2, 3, 4])
    validElementsPerSample = (
        (targetMask * lossWeighting).sum(dim=[1, 2, 3, 4]).clamp_min(1.0)
    )

    meanLossPerSample = lossPerSample / validElementsPerSample
    loss = meanLossPerSample.mean()

    return loss
