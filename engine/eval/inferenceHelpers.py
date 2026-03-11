import torch 
from engine.train.diffusion import EDMNoiseScheduler
from engine.train.builder import LatentSequenceBuilder
from cosmos_predict2._src.predict2.cosmos_policy.modules.cosmos_sampler import CosmosPolicySampler
from omegaconf import DictConfig, OmegaConf

"""
Inference Solver Class 
"""

class CosmosInferenceSolver:
    def __init__(self,cfg:DictConfig,diffusionModel,device):
        self.diffusionModel = diffusionModel
        self.scheduler = CosmosPolicySampler()
        self.device = device 
        self.noiseScheduler = EDMNoiseScheduler(
            cfg=cfg,
            inference=True,
            device=device
        )
        
        noiseSchedulerCFG = cfg.inference.noiseScheduler
        self.sigmaMax = noiseSchedulerCFG.sigmaMax
        self.sigmaMin = noiseSchedulerCFG.sigmaMin
        self.sigmaData = noiseSchedulerCFG.sigmaData
        self.rho = noiseSchedulerCFG.rho
    
    @torch.no_grad()
    #there is no CFG for cosmos policy 
    def denoiser(self,noisyNetworkInput,sigma,crossAttentionEmbed,conditioningMasks):
        batchSize= noisyNetworkInput.shape[0]
        T = noisyNetworkInput.shape[2]
        
        skipSigma, outputSigma, cinSigma, noiseSigma, _ = (
            self.noiseScheduler.EDMScalingFactors(sigma)
        )
        sigma = sigma.view(batchSize, 1, 1, 1, 1)
        skipSigma = skipSigma.view(batchSize, 1, 1, 1, 1)
        outputSigma = outputSigma.view(batchSize, 1, 1, 1, 1)
        cinSigma = cinSigma.view(batchSize, 1, 1, 1, 1)
        noiseSigma = noiseSigma.view(batchSize, 1).repeat(1, T)
        
        maskedXSigma = noisyNetworkInput
        networkInput = maskedXSigma * cinSigma

        
        #padding mask 
        B, C, T, H, W = noisyNetworkInput.shape
        paddingMask = torch.zeros(
            B,1,H,W,device=self.device,dtype=torch.bfloat16,  
        )
        
        
        # modeloutput
        networkOutput = self.diffusionModel.forward(
            latent=networkInput,
            timesteps=noiseSigma,
            crossAttentionEmbed=crossAttentionEmbed,
            conditionVideoMask=conditioningMasks,
            paddingMask=paddingMask
        )

        denoisedPrediction = (maskedXSigma * skipSigma) + (networkOutput * outputSigma)
        
        return denoisedPrediction
        

    def runSolver(self,builtVAEInput,crossAttentionEmbed,conditioningMasks,numSteps):
        pureNoise = torch.randn_like(builtVAEInput)
        noisyInput = conditioningMasks * builtVAEInput + (1-conditioningMasks) * pureNoise
        
        def _denoiser(x,sigma):
            return self.denoiser(
                x,sigma,crossAttentionEmbed,conditioningMasks
            )
        
        return self.scheduler(
            x0_fn=_denoiser,
            x_sigma_max=noisyInput,
            num_steps=numSteps,
            sigma_min=self.sigmaMin,
            sigma_max=self.sigmaMax,
            rho=self.rho, #default
            solver_option="2ab"
        )
        
#Action prediction builder
class InferenceActionBuilder(LatentSequenceBuilder):
    def __init__(self, cfg, vaeModel, device):
        super().__init__(cfg, vaeModel, device)
    
    def _normalizeProprio(self,currentProprio):
        eps = 1e-8
        currentPropScaled = (currentProprio - self.propMin) / (
            (self.propMax - self.propMin) + eps  # pyrefly:ignore
        )
        currentPropNorm = (2.0 * currentPropScaled) - 1.0
        currentPropNorm = torch.clamp(currentPropNorm, min=-1.0, max=1.0)
        
        return currentPropNorm
    
    @torch.no_grad()
    def forward( #pyrefly:ignore 
        self,
        currentProprio,
        currentWristImg,
        currentLeftImg,
        currentRightImg,
        collectedActions=None,     
        futureProprio=None,        
        futureWristImg=None,        
        futureLeftImg=None,         
        futureRightImg=None,        
        futureValue=None,           
    ):
        currentWristImg = self._rescaleImg(currentWristImg)
        currentLeftImg = self._rescaleImg(currentLeftImg)
        currentRightImg = self._rescaleImg(currentRightImg)
        
        #rescale proprio 
        rescaledCurrentProprio = self._normalizeProprio(currentProprio)
        
        blankPlaceHolder = torch.zeros_like(currentWristImg) 
        
        inputToVAE = self._buildVAEInput(
            currentWristImg=currentWristImg,
            currentLeftImg=currentLeftImg,
            currentRightImg=currentRightImg,
            futureWristImg=blankPlaceHolder,
            futureLeftImg=blankPlaceHolder,
            futureRightImg=blankPlaceHolder,
        ).to(torch.bfloat16)
        
        vaeOutput = self._VAEForward(inputToVAE)
        reshapedCurrentProp = self._latentBlockMaker(rescaledCurrentProprio)
        
        vaeOutput = vaeOutput.clone()
        vaeOutput[:, :, 1, :, :] = reshapedCurrentProp.to(vaeOutput.dtype)
        
        return vaeOutput
        
#Future State Builder
class InferenceWMBuilder(LatentSequenceBuilder):
    
    def __init__(self, cfg, vaeModel, device):
        super().__init__(cfg, vaeModel, device)
        
    def _normalizeProprioAction(self,currentProprio,collectedActions):
        eps = 1e-8
        currentPropScaled = (currentProprio - self.propMin) / (
            (self.propMax - self.propMin) + eps  # pyrefly:ignore
        )
        currentPropNorm = (2.0 * currentPropScaled) - 1.0
        currentPropNorm = torch.clamp(currentPropNorm, min=-1.0, max=1.0)
        actionsScaled = (collectedActions - self.actMin) / (
            (self.actMax - self.actMin) + eps  # pyrefly:ignore
        )
        actionsNorm = (2.0 * actionsScaled) - 1.0
        actionsNorm = torch.clamp(actionsNorm, min=-1.0, max=1.0)
        return currentPropNorm,actionsNorm
    
    @torch.no_grad()
    def forward( #pyrefly:ignore 
        self,
        currentProprio,
        currentWristImg,
        currentLeftImg,
        currentRightImg,
        collectedActions,     
        futureProprio=None,        
        futureWristImg=None,        
        futureLeftImg=None,         
        futureRightImg=None,        
        futureValue=None,           
    ):
        currentWristImg = self._rescaleImg(currentWristImg)
        currentLeftImg = self._rescaleImg(currentLeftImg)
        currentRightImg = self._rescaleImg(currentRightImg)
        
        #rescale proprio 
        rescaledCurrentProprio,actionsNorm = self._normalizeProprioAction(currentProprio,collectedActions)
        
        blankPlaceHolder = torch.zeros_like(currentWristImg) 
        
        inputToVAE = self._buildVAEInput(
            currentWristImg=currentWristImg,
            currentLeftImg=currentLeftImg,
            currentRightImg=currentRightImg,
            futureWristImg=blankPlaceHolder,
            futureLeftImg=blankPlaceHolder,
            futureRightImg=blankPlaceHolder,
        ).to(torch.bfloat16)
        
        vaeOutput = self._VAEForward(inputToVAE)

        reshapedCurrentProp = self._latentBlockMaker(rescaledCurrentProprio)

        reshapedActions = self._latentBlockMaker(actionsNorm)
        vaeOutput = vaeOutput.clone()
        vaeOutput[:, :, 1, :, :] = reshapedCurrentProp.to(vaeOutput.dtype)
        vaeOutput[:, :, 5, :, :] = reshapedActions.to(vaeOutput.dtype)

        return vaeOutput
        
        

#Value Function Builder
class InferenceValueBuilder(LatentSequenceBuilder):
    def __init__(self, cfg, vaeModel, device):
        super().__init__(cfg, vaeModel, device)
    
    @torch.no_grad()
    def forward( #pyrefly:ignore 
        self,
        currentProprio,
        currentWristImg,
        currentLeftImg,
        currentRightImg,
        collectedActions,     
        futureProprio,        
        futureWristImg,        
        futureLeftImg,         
        futureRightImg,        
        futureValue=None,           
    ):
        currentWristImg = self._rescaleImg(currentWristImg)
        currentLeftImg = self._rescaleImg(currentLeftImg)
        currentRightImg = self._rescaleImg(currentRightImg)
        futureWristImg = self._rescaleImg(futureWristImg)
        futureLeftImg = self._rescaleImg(futureLeftImg)
        futureRightImg = self._rescaleImg(futureRightImg)
        
        currentPropNorm, actionsNorm, futurePropNorm = self._normalizePhysical(
            currentProprio,
            collectedActions,
            futureProprio,
        )
        
        inputToVAE = self._buildVAEInput(
            currentWristImg=currentWristImg,
            currentLeftImg=currentLeftImg,
            currentRightImg=currentRightImg,
            futureWristImg=futureWristImg,
            futureLeftImg=futureLeftImg,
            futureRightImg=futureRightImg,
        ).to(torch.bfloat16)

        vaeOutput = self._VAEForward(inputToVAE)

        reshapedCurrentProp = self._latentBlockMaker(currentPropNorm)

        reshapedActions = self._latentBlockMaker(actionsNorm)

        reshapedFutureProp = self._latentBlockMaker(futurePropNorm)
        vaeOutput = vaeOutput.clone()
        vaeOutput[:, :, 1, :, :] = reshapedCurrentProp.to(vaeOutput.dtype)
        vaeOutput[:, :, 5, :, :] = reshapedActions.to(vaeOutput.dtype)
        vaeOutput[:, :, 6, :, :] = reshapedFutureProp.to(vaeOutput.dtype)
        
        return vaeOutput
        
        
