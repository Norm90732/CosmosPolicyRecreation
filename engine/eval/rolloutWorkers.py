import ray 
from omegaconf import DictConfig
import h5py 
import numpy as np 
from PIL import Image
from pathlib import Path 
import io 
import torch 
import gc 
import pickle 
from engine.eval.inferenceHelpers import (
    CosmosInferenceSolver,
    InferenceActionBuilder,
    unNormalizeLatents
)
from models.cosmosLoaderBase import loadCosmosModules, EncoderVAE,CosmosDiffusionNet
from models.cosmosLoaderTrained import loadTrainedPolicyModel

@ray.remote(num_cpus=1,num_gpus=1)
class CosmosInferencePolicyServer:
    def __init__(self,cfg:DictConfig):
        self.device = torch.device("cuda")
        self.cfg = cfg 
        self.actionHorizon = cfg.inference.actionHorizon
        print("Loading Cosmos Model")
        vae, textEncoder, net, config = loadCosmosModules(cfg)
        
        policyNet = loadTrainedPolicyModel(cfg,net)
        print("Loaded policyNet weights")
        
        modelVAE = EncoderVAE(vae)
        policyNetModel = CosmosDiffusionNet(policyNet)
        
        del textEncoder
        torch.cuda.empty_cache()
        gc.collect()
        print("All models loaded")
        #load text embeddings onto gpu 
        embKeys = np.load(Path(cfg.dataset.reasonMMAP, "embeddingkeys.npy"), allow_pickle=True)
        embValues = np.load(Path(cfg.dataset.reasonMMAP, "embeddingvalues.npy"))  

        self.embIndex = {k: i for i, k in enumerate(embKeys)}
        self.embValues = torch.from_numpy(embValues).float().to(self.device)
        
        #define solver, action builder, unnormalize latents
        
        self.inferenceSolver = CosmosInferenceSolver(
            cfg=cfg,diffusionModel=policyNetModel,device=self.device
        )
        
        self.infereceActionBuilder = InferenceActionBuilder(
            cfg=cfg,vaeModel=modelVAE,device=self.device
        )
        
        self.unNormalizeLatents = unNormalizeLatents(cfg)
        
        
        
        
    def _getEmbedding(self, textString: str) -> torch.Tensor:
        idx = self.embIndex[textString]
        return self.embValues[idx]  
        
    def _buildActionMask(self,vaeInput:torch.Tensor,T:int=11,H:int=28,W:int=28):
        
        B = vaeInput.shape[0]

        conditionVideoMask = torch.zeros(
            (B, 1, T, H, W), device=self.device, dtype=torch.bfloat16
        )
        
        conditionVideoMask[:, :, :5, :, :] = (1.0)
        
        
        return conditionVideoMask
    
    
    def predictionActions(self,)
        
        
        
        
        
        






@ray.remote(num_cpus=1)
class HDF5Saver():
    def __init__(self,cfg:DictConfig,saveDir:str):
        self.cfg = cfg 
        self.saveDir = Path(saveDir)
        self.saveDir.mkdir(parents=True,exist_ok=True)
        
    def _jpegDecode(self,jpegBytes: bytes):
        buf = io.BytesIO(jpegBytes.tobytes()) #pyrefly:ignore 
        return np.array(Image.open(buf))  
    
    def _jpegEncode(self,image):
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="JPEG", quality=95)
        return np.frombuffer(buf.getvalue(), dtype=np.uint8)
    
    def saveEpisode(self,exportDict: dict, fileName: str) -> str:
        savePath = self.saveDir / fileName
        T = exportDict["primary_images"].shape[0]
        dt = h5py.vlen_dtype(np.dtype("uint8")) 

        with h5py.File(savePath, "w") as f:
            f.attrs["task_description"] = exportDict["task_description"]
            f.attrs["success"] = bool(exportDict["success"])

            f.create_dataset("actions", data=exportDict["actions"])  
            f.create_dataset("proprio", data=exportDict["proprio"]) 

            for h5Key, dictKey in [
                ("primary_images_jpeg",   "primary_images"),
                ("secondary_images_jpeg", "secondary_images"),
                ("wrist_images_jpeg",     "wrist_images"),
            ]:
                ds = f.create_dataset(h5Key, shape=(T,), dtype=dt)
                for t in range(T):
                    ds[t] = self._jpegEncode(exportDict[dictKey][t])
                    
        return str(savePath)