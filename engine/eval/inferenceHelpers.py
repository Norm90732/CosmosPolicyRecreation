import torch 
from engine.train.diffusion import EDMNoiseScheduler
from engine.train.builder import LatentSequenceBuilder
from cosmos_predict2._src.predict2.cosmos_policy.modules.cosmos_sampler import CosmosPolicySampler
from omegaconf import DictConfig, OmegaConf
from einops import rearrange
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
            predictedX0 =  self.denoiser(x,sigma,crossAttentionEmbed,conditioningMasks)
            
            lockedCondition = conditioningMasks * builtVAEInput + (1-conditioningMasks) * predictedX0
            
            return lockedCondition
        
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

def worldModelInjector(originalVAELatent:torch.Tensor,predictedActionChunk:torch.Tensor):
    originalVAELatent = originalVAELatent.clone()
    originalVAELatent[:, :, 5, :, :] = predictedActionChunk.to(originalVAELatent.dtype)
    return originalVAELatent

def valueModelInjector(originalVAELatent:torch.Tensor,predictedActionChunk:torch.Tensor,
                       predictedFutureWristImg:torch.Tensor,predictedFutureLeftImg:torch.Tensor,
                       predictedFutureRightImg:torch.Tensor,predictedFutureProprio):
    originalVAELatent = originalVAELatent.clone()
    originalVAELatent[:, :, 5, :, :] = predictedActionChunk.to(originalVAELatent.dtype)
    originalVAELatent[:, :, 6, :, :] = predictedFutureProprio.to(originalVAELatent.dtype)
    originalVAELatent[:, :, 7, :, :] = predictedFutureWristImg.to(originalVAELatent.dtype)
    originalVAELatent[:, :, 8, :, :] = predictedFutureLeftImg.to(originalVAELatent.dtype)
    originalVAELatent[:, :, 9, :, :] = predictedFutureRightImg.to(originalVAELatent.dtype)
    return originalVAELatent
    

class unNormalizeLatents(torch.nn.Module):
    """
    Averages Latent -> Un Normalizes -> Returns Proper Shape for Robot 
    """
    def __init__(self,cfg:DictConfig,actionHorizon:int=32,actionDim:int=7,propDim:int=9):
        super().__init__()
        import json 
        self.imageSize = cfg.dataset.imageSize
        self.latentSize = self.imageSize // 8
        self.actionDim = actionDim
        self.actionHorizon= actionHorizon
        self.propDim = propDim
        statisticsPath = cfg.dataset.normalizationFile
        with open(statisticsPath, "r") as f:
            stats = json.load(f)
        
        self.register_buffer(
            "actMin", torch.tensor(stats["actions_min"], dtype=torch.float32)
        )
        self.register_buffer(
            "actMax", torch.tensor(stats["actions_max"], dtype=torch.float32)
        )

        self.register_buffer(
            "propMin", torch.tensor(stats["proprio_min"], dtype=torch.float32)
        )
        self.register_buffer(
            "propMax", torch.tensor(stats["proprio_max"], dtype=torch.float32)
        )
    
    def _inverseLatentBlock(self,latent,flattenedLength):
        desiredSize = 16 * self.latentSize * self.latentSize
        
        flattened = torch.flatten(latent,1)
        
        fullCopies = desiredSize // flattenedLength
        
        recovered = flattened[:, : fullCopies * flattenedLength]
        recovered = rearrange(recovered, "b (n r) -> b n r", n=fullCopies, r=flattenedLength)
        recovered = recovered.mean(dim=1)
        
        return recovered
    def unnormAction(self,actionChunk):
        actionDim = self.actionDim
        flattenedLength = self.actionHorizon * actionDim
        
        flat = self._inverseLatentBlock(actionChunk, flattenedLength)
        actNorm = (flat + 1.0) / 2.0                         
        actions = actNorm * (self.actMax - self.actMin) + self.actMin #pyrefly:ignore 
        actions = rearrange(actions,"b (t d) -> b t d", t = self.actionHorizon,d = self.actionDim)
        return actions 
    def unnormProp(self,propChunk):
        flat = self._inverseLatentBlock(propChunk,self.propDim)
        propNorm = (flat + 1.0) / 2.0
        proprio = propNorm * (self.propMax - self.propMin) + self.propMin #pyrefly:ignore 

        return proprio 
    
    def unnormValue(self,valueLatent):
        flat = self._inverseLatentBlock(valueLatent, flattenedLength=1)
        value = (flat + 1.0) / 2.0

        return value
    
    @torch.no_grad()
    def forward(self,vaeOutput):
        actions  = self.unnormAction(vaeOutput[:, :, 5, :, :])
        currProp = self.unnormProp(vaeOutput[:, :, 1, :, :])
        futureProp = self.unnormProp(vaeOutput[:, :, 6, :, :])
        value    = self.unnormValue(vaeOutput[:, :, 10, :, :])
        
        return {
            "actions": actions,          
            "currentProprio": currProp,  
            "futureProprio": futureProp, 
            "value": value,             
        }
       