from jaxtyping import jaxtyped,Float
import torch
from torch import Tensor 
from omegaconf import DictConfig
from diffusers import AutoencoderKLWan
from beartype import beartype


class EncoderVAE(torch.nn.Module):
    def __init__(self,device):
        super().__init__()
        
        self.VAE = AutoencoderKLWan.from_pretrained( #pyrefly:ignore 
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to(device) 
        #Cosmos Normalization Values 
        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        
        self.register_buffer("mean", torch.tensor(mean).reshape(1, 16, 1, 1, 1))
        self.register_buffer("std",  torch.tensor(std).reshape(1, 16, 1, 1, 1))
        
        self.VAE.requires_grad_(False)
        self.VAE.eval()
            
    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "b rgb t h w"]) -> Float[Tensor, "b c latentT latentH latentW"]:
        dist = self.VAE.encode(x).latent_dist
        latents = dist.mode()   
        return (latents - self.mean.type_as(latents)) / self.std.type_as(latents) #pyrefly:ignore 
#(B, 16, 1 + (T-1)//4, H//8, W//8)
        
        
    
    