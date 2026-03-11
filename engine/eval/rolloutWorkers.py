import ray 
from omegaconf import DictConfig
import h5py 
import numpy as np 
from PIL import Image
from pathlib import Path 
import io 
from engine.eval.inferenceHelpers import (
    CosmosInferenceSolver,
    InferenceActionBuilder,
    unNormalizeLatents
)

@ray.remote(num_cpus=1,num_gpus=0.1)







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