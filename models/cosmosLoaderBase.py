# pyrefly: ignore-errors
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_predict2._src.predict2.conditioner import DataType
from omegaconf import DictConfig, OmegaConf
import torch
from jaxtyping import jaxtyped, Float
from torch import Tensor
from beartype import beartype
from typing import Optional


# base loader 
def loadCosmosModules(cfg: DictConfig):
    model, config = load_model_from_checkpoint(
        experiment_name="Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2",
        s3_checkpoint_dir=cfg.model.ckptPath,
        config_file="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        load_ema_to_reg=True,
        experiment_opts=["~data_train"],
    )
    
            
    vae = model.tokenizer
    textEncoder = model.text_encoder
    net = model.net

    del model
    torch.cuda.empty_cache()

    return vae, textEncoder, net, config


"""
Wan Video 2.1 VAE 

#(B, 16, 1 + (T-1)//4, H//8, W//8)
"""

class EncoderVAE(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "b rgb t h w"]
    ) -> Float[Tensor, "b 16 latentT latentH latentW"]:
        return self.vae.encode(x)

class DecoderVAE(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def forward(
        self, z: Float[Tensor, "b c tLatent hLatent wLatent"]
    ) -> Float[Tensor, "b rgb t hReal wReal"]:
        return self.vae.decode(z)
    
"""
Cosmos Reason 1 Text Encoder 

Embed Dim = 100352 
"""


class ReasonTextEncoder(torch.nn.Module):
    def __init__(self, textEncoder):
        super().__init__()
        self.textEncoder = textEncoder

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def forward(
        self, prompt: str
    ) -> Float[Tensor, "1 seqlen embedDim"]:  # 100352 = embedDim
        out = self.textEncoder.compute_text_embeddings_online(
            data_batch={"ai_caption": [prompt], "images": None},
            input_caption_key="ai_caption",
        )
        return out


"""
Cosmos Diffusion BackBone Model 

Model Being Fine Tuned for Policy and World Model. 

Forward Args: 

latent: 

Latent video tensor from VAE latent encoding and latent injection input. 

timesteps:

EDM sigma noise time steps from sampling

crossAttentionEmbed:

Cross attention conditioning from Cosmos-Reason1 text encoder

conditionVideoMask:

Binary mask to mark conditioning frames vs frames to be generated. 

1 = condition frame
0=frame to generate
[B, 1, T, H, W]

fps:

Framerate of input video 

Input is a scalar. 

paddingMask:
Spatial padding mask at B, H, W 
0 = valid region, 1 = padded region 

dataType:

Data selector Video or Image can be passed in 

imgContextEmbed:

Image to Video tasks, inject static image representation. 
"""


class CosmosDiffusionNet(torch.nn.Module):
    def __init__(self, net):
        super().__init__()

        self.net = net

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        latent: Float[Tensor, "b c t h w"],
        timesteps: Float[Tensor, "b time"],
        crossAttentionEmbed: Float[Tensor, "b seq reasonEmbedDim"],
        conditionVideoMask: Float[Tensor, "b 1 t h w"],
        fps: Optional[torch.Tensor] = None,
        paddingMask: Optional[torch.Tensor] = None,
        dataType: DataType = DataType.VIDEO,
        imgContextEmbed: Optional[torch.Tensor] = None,
    ) -> Float[Tensor, "b c t h w"]:

        latent = latent.to(torch.bfloat16)
        timesteps = timesteps.to(torch.bfloat16)
        crossAttentionEmbed = crossAttentionEmbed.to(torch.bfloat16)
        conditionVideoMask = conditionVideoMask.to(torch.bfloat16)
        if paddingMask is not None:
            paddingMask = paddingMask.to(torch.bfloat16)
        
        
        return self.net(
            x_B_C_T_H_W=latent,
            timesteps_B_T=timesteps,
            crossattn_emb=crossAttentionEmbed,
            padding_mask=paddingMask,
            fps=fps,
            data_type=dataType,
            img_context_emb=imgContextEmbed,
            condition_video_input_mask_B_C_T_H_W=conditionVideoMask,
        )


"""
if __name__ == "__main__":
    cfg = OmegaConf.load("configs/config.yaml")
    modelCfg = OmegaConf.load("configs/model/policy.yaml")
    cfg.model = modelCfg
    
    vae, text_encoder, net, config = loadCosmosModules(cfg)
    
    vaeEncoder  = EncoderVAE(vae)
    textEncoder = ReasonTextEncoder(text_encoder)
    diffNet     = CosmosDiffusionNet(net)
    
    print("Loaded all the models ")
"""