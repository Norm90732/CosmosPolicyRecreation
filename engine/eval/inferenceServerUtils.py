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
    
    #selfLatentBuilder 
    
    @torch.no_grad()
    #there is no CFG for cosmos policy 
    def denoiser(self,noisyNetworkInput,sigma,crossAttentionEmbed,conditionVideoMask,paddingMask):
        batchSize= noisyNetworkInput.shape[0]
        T = noisyNetworkInput.shape[2]
        sigma = self.noiseScheduler.sampleSigma(batchSize, device=self.device)
        
        skipSigma, outputSigma, cinSigma, noiseSigma, _ = (
            self.noiseScheduler.EDMScalingFactors(sigma)
        )
        sigma = sigma.view(batchSize, 1, 1, 1, 1)
        skipSigma = skipSigma.view(batchSize, 1, 1, 1, 1)
        outputSigma = outputSigma.view(batchSize, 1, 1, 1, 1)
        cinSigma = cinSigma.view(batchSize, 1, 1, 1, 1)
        noiseSigma = noiseSigma.view(batchSize, 1).repeat(1, T)
        epsilon = torch.randn_like(noisyNetworkInput)
        
    def runSolver():
        pass 
    

#Action prediction builder
class InferenceActionBuilder(LatentSequenceBuilder):
    def __init__(self, cfg, vaeModel, device):
        super().__init__(cfg, vaeModel, device)
    
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
        pass 


#Future State Builder
class InferenceWMBuilder(LatentSequenceBuilder):
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
        futureProprio=None,        
        futureWristImg=None,        
        futureLeftImg=None,         
        futureRightImg=None,        
        futureValue=None,           
    ):
        pass 

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
        futureValue,           
    ):
        pass 
