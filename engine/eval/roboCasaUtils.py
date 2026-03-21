#rendering HDF5 files 
from models.cosmosLoaderTrained import loadTrainedPolicyModel
from models.cosmosLoaderBase import loadCosmosModules, DecoderVAE
import numpy as np 
import h5py
import numpy as np 
import io
from PIL import Image 
import torch 
from einops import rearrange
from pathlib import Path 
def renderRobocasa(decoderVAE,rolloutPath:str, gifName:str,saveDir:str,device:torch.device) -> None:
    #open h5py and extract all paths, render new path sequentially in vae decoder, then construct gif
    #open bytes 
    def decodeJpeg(frameData):
        return Image.open(io.BytesIO(bytes(frameData)))
    #unnormalize 
    def latentToImage(decodedTensor):
        img = decodedTensor.clamp(-1, 1)
        img = (img + 1.0) / 2.0 * 255.0
        img = img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return img 
    
    with h5py.File(rolloutPath, 'r') as f:
        leftData = f['primary_images_jpeg'][:]
        wristImg = f['wrist_images_jpeg'][:]
        rightImg = f['secondary_images_jpeg'][:]
        
        #latent 
        leftLatent = f["predicted_primary_latents"][:]
        wristLatent = f["predicted_wrist_latents"][:]
        rightLatent = f["predicted_secondary_latents"][:]

    numFrames = min(len(leftData), len(wristImg), len(rightImg))
    numLatents = len(leftLatent)  
    framesPerChunk = numFrames // numLatents 
    
    leftLatentTensor = torch.from_numpy(leftLatent).float().to(device)
    wristLatentTensor = torch.from_numpy(wristLatent).float().to(device)
    rightLatentTensor = torch.from_numpy(rightLatent).float().to(device)
    
    leftLatentTensor = rearrange(leftLatentTensor,"b c h w -> b c 1 h w")
    wristLatentTensor = rearrange(wristLatentTensor,"b c h w -> b c 1 h w")
    rightLatentTensor = rearrange(rightLatentTensor,"b c h w -> b c 1 h w")
    
    with torch.no_grad():
        decodedLeft = decoderVAE.forward(leftLatentTensor)
        decodedWrist = decoderVAE.forward(wristLatentTensor)
        decodedRight = decoderVAE.forward(rightLatentTensor)
    #remove t
    decodedLeft = decodedLeft.squeeze(2)
    decodedWrist = decodedWrist.squeeze(2)
    decodedRight = decodedRight.squeeze(2)
    
    leftImgsDecoded = latentToImage(decodedLeft)  
    wristImgsDecoded = latentToImage(decodedWrist)
    rightImgsDecoded = latentToImage(decodedRight)
    
    
    frames = []
    for i in range(numFrames):
        realLeft = decodeJpeg(leftData[i]).resize((224, 224))
        realWrist = decodeJpeg(wristImg[i]).resize((224, 224))
        realRight = decodeJpeg(rightImg[i]).resize((224, 224))
        
        latentIdx = min(i // framesPerChunk, numLatents - 1)
        hallLeft = Image.fromarray(leftImgsDecoded[latentIdx]).resize((224, 224))
        hallWrist = Image.fromarray(wristImgsDecoded[latentIdx]).resize((224, 224))
        hallRight = Image.fromarray(rightImgsDecoded[latentIdx]).resize((224, 224))
        
        combined = Image.new('RGB', (224 * 3, 224 * 2))
        combined.paste(realLeft,  (0, 0))
        combined.paste(realWrist, (224, 0))
        combined.paste(realRight, (448, 0))
        combined.paste(hallLeft,  (0, 224))
        combined.paste(hallWrist, (224, 224))
        combined.paste(hallRight, (448, 224))
        
        frames.append(combined)
        
        
    savePath = Path(saveDir) / gifName
    frames[0].save(
        savePath,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / 10),
        loop=0
    )
    print(f"Saved to {savePath}")
            
    
    
#clean up later, but with recflow trained model. 
"""
if __name__ == "__main__":
    import gc 
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/config.yaml")
    cfg.dataset = OmegaConf.load("configs/dataset/robocasaRollout.yaml")
    cfg.model = OmegaConf.load("configs/model/policy.yaml")
    cfg.inference = OmegaConf.load("configs/inference/rollout.yaml")
    
    vae, textEncoder, net, config = loadCosmosModules(cfg) #pyrefly:ignore 
    modelVAE = DecoderVAE(vae)
    
    del textEncoder
    del net
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda")
    
    saveDir = "/home/normansmith/blue_storage/projects/CosmosPolicyRecreation"
    renderRobocasa(modelVAE,rolloutPath="/home/normansmith/blue_storage/projects/CosmosPolicyRecreation/task=OpenDoubleDoor--ep=40--success=True.hdf5", gifName="openDoubleDoor.gif",saveDir=saveDir,device=device)
    renderRobocasa(modelVAE,rolloutPath="/home/normansmith/blue_storage/projects/CosmosPolicyRecreation/task=PnPCounterToCab--ep=12--success=True.hdf5", gifName="counterToCabinet.gif",saveDir=saveDir,device=device)
    renderRobocasa(modelVAE,rolloutPath="/home/normansmith/blue_storage/projects/CosmosPolicyRecreation/task=TurnSinkSpout--ep=40--success=True.hdf5", gifName="TurnSinkSpout.gif",saveDir=saveDir,device=device)
"""
    