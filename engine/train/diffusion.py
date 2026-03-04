from cosmos_predict2._src.predict2.models.fm_solvers_unipc import (
    FlowUniPCMultistepScheduler,
)
import torch
from omegaconf import DictConfig
from torch import Tensor

"""
Define noise scheduler for training and inference
"""


class EDMNoiseScheduler:
    def __init__(self, cfg: DictConfig, inference: bool, device=torch.device):
        if inference == True:
            noiseSchedulerCFG = cfg.inference.noiseScheduler
        else:
            noiseSchedulerCFG = cfg.model.noiseScheduler

        self.device = device
        self.pMean = noiseSchedulerCFG.pMean
        self.pSTD = noiseSchedulerCFG.pSTD
        self.sigmaMax = noiseSchedulerCFG.sigmaMax
        self.sigmaMin = noiseSchedulerCFG.sigmaMin
        self.sigmaData = noiseSchedulerCFG.sigmaData
        self.uniformLower = noiseSchedulerCFG.uniformLower
        self.uniformUpper = noiseSchedulerCFG.uniformUpper
        self.logNormalSampleProb = noiseSchedulerCFG.logNormalSampleProb
        self.uniformSampleProb = noiseSchedulerCFG.uniformSampleProb

        # logNormal Definition
        self.normalDistribution = torch.distributions.Normal(
            loc=self.pMean, scale=self.pSTD
        )
        # Uniform Distribution
        self.uniformDistribution = torch.distributions.Uniform(
            low=self.uniformLower, high=self.uniformUpper
        )

    def sampleSigma(self, batch: int, device: torch.device) -> Tensor:
        # paper uses [0.7,0.3 split sample]
        mask = torch.rand(batch, device=device) < self.logNormalSampleProb

        # log normal
        p = self.normalDistribution.sample((batch,)).to(device)
        sigmaLogNormal = torch.exp(p).clamp(self.sigmaMin, self.sigmaMax)

        # uniform
        sigmaUniform = self.uniformDistribution.sample((batch,)).to(device)
        sigmaUniform = sigmaUniform.clamp(self.sigmaMin, self.sigmaMax)

        sigma = torch.where(mask, sigmaLogNormal, sigmaUniform)
        return sigma

    def EDMScalingFactors(
        self, sigma: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = sigma.shape[0]
        unitSigma = sigma**2 + self.sigmaData**2

        cinSigma = 1 / (torch.sqrt(unitSigma))
        noiseSigma = 0.25 * torch.log(sigma)

        outputSigma = (sigma * self.sigmaData) / torch.sqrt(unitSigma)

        skipSigma = (self.sigmaData**2) / (unitSigma)

        lossWeighting = unitSigma / (sigma * self.sigmaData) ** 2

        return skipSigma, outputSigma, cinSigma, noiseSigma, lossWeighting


def buildInferenceScheduler(
    device: torch.device, numSteps: int
) -> FlowUniPCMultistepScheduler:
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        solver_order=2,
        prediction_type="flow_prediction",
        solver_type="bh2",
        shift=1,
        use_dynamic_shifting=False,
        lower_order_final=True,
        final_sigmas_type="zero",
        predict_x0=True,
    )

    scheduler.set_timesteps(
        num_inference_steps=numSteps,
        device=device,
        shift=5,  # RoboCasa shift
    )

    return scheduler
