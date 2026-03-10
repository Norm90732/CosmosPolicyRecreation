import torch 
from omegaconf import DictConfig

def loadTrainedPolicyModel(cfg:DictConfig,net):
    checkpointPath = cfg.inference.checkpoints.get("actionModelCheckpoint", None)
    
    if not checkpointPath:
        raise ValueError("cfg.inference.checkpoints.actionModelCheckpoint is not set")
    raw = torch.load(checkpointPath, map_location="cpu")
    state = raw["model"] 
    #remove .net from model 
    cleanedState = {}
    for k, v in state.items():
        if k.startswith("net."):
            newK = k[len("net.") :]
        else:
            newK = k
        cleanedState[newK] = v

    net.load_state_dict(cleanedState, strict=True)
    
    print("Loaded policy model")
    return net  
