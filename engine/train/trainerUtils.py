import torch 
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR,SequentialLR, LinearLR,ConstantLR
from torch import Tensor 


"""
conditioningMasks = {
            "policy":         [0,1, 2, 3, 4], # s
            "worldModel":    [0,1, 2, 3, 4, 5], # s + a
            "valueFunction": [0,1, 2, 3, 4, 5, 6, 7, 8, 9], # s + a + s'
        }
"""
def buildConditioningMasks(isDemo:torch.Tensor,device:torch.device,stage:int,valueFunctionVariant: str="base",T: int = 11,H:int=28,W:int=28) -> Tensor:
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
    
    conditionVideoMask[:, :, :5, :, :] = 1.0    #base that applies to policy, world model, and value function 
    
    randomValues = torch.rand(B,device=device)
    
    #paper uses a 50 | 25, 25 split for stage 1
    if stage == 1: 
        isPolicy = isDemo
        
        isWorldModel = (~isDemo) & (randomValues < 0.5) #mask 
        
        isValueFunction = (~isDemo) & (randomValues >= 0.5)
        
    #paper uses a 10 | 45, 45 split for stage 2     
    elif stage ==2:
        #isDemo = false 
        isPolicy        = randomValues < 0.10
        isWorldModel    = (randomValues >= 0.10) & (randomValues < 0.55)
        isValueFunction = randomValues >= 0.55
        
   
    conditionVideoMask[isWorldModel, :, 5, :, :]= 1.0 #pyrefly:ignore
    
    if valueFunctionVariant == "base":
        #condition on full sequence to predict the value
        conditionVideoMask[isValueFunction, :, 5:10, :, :] = 1.0 #pyrefly:ignore 

    elif valueFunctionVariant == "Vs":
        #condition on s'
        conditionVideoMask[isValueFunction, :, :6, :, :] = 0.0 #pyrefly:ignore 
        conditionVideoMask[isValueFunction, :, 6:10, :, :] = 1.0 #pyrefly:ignore 
        
    elif valueFunctionVariant == "Qsa": 
        #condition on s,a, predict s' 
        conditionVideoMask[isValueFunction, :, :6, :, :] = 1.0 #pyrefly:ignore 
        conditionVideoMask[isValueFunction, :, 6:10, :, :] = 0.0 #pyrefly:ignore 
               
    return conditionVideoMask




def optimizerAndSchedulerCreator(model, cfg: DictConfig) -> tuple[Adam, SequentialLR]:
    cfgOptimizer = cfg.model.training.optimizer
    cfgScheduler = cfg.model.training.scheduler

    optimizer = Adam(
        model.parameters(),
        lr=cfgOptimizer.lr,
        betas=tuple(cfgOptimizer.betas), #pyrefly:ignore 
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
        eta_min=cfgScheduler.cosine.etaMin,
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