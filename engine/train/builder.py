import torch
from omegaconf import DictConfig
import hydra
from models.cosmosLoaderBase import EncoderVAE
import json
from einops import rearrange, repeat
from torch import Tensor
import math

"""
Buiilder of Latent Sequence
"""


class LatentSequenceBuilder(torch.nn.Module):
    def __init__(self, cfg: DictConfig, vaeModel, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.vae = vaeModel

        self.imageSize = cfg.dataset.imageSize
        self.latentSize = self.imageSize // 8

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

    def _repeatImageFourTimes(self, imgTensor: Tensor) -> Tensor:
        # B, C, H, W
        imgTensor = rearrange(imgTensor, "b c h w -> b c 1 h w")
        imgTensor = repeat(imgTensor, "b c t h w -> b c (four t) h w", four=4)
        return imgTensor

    def _buildVAEInput(
        self,
        currentWristImg,
        currentLeftImg,
        currentRightImg,
        futureWristImg,
        futureLeftImg,
        futureRightImg,
    ) -> Tensor:
        """
        First 5 elements are s state
        """
        # first part of sequence
        blankPlaceHolder = torch.zeros_like(currentWristImg)  # B, C, H, W

        # second part of sqeuence
        currentProprioPlaceholder = self._repeatImageFourTimes(blankPlaceHolder)

        # third part of sequence
        currentWristImgRepeated = self._repeatImageFourTimes(currentWristImg)

        # fourth part of sequence
        currentLeftImgRepeated = self._repeatImageFourTimes(currentLeftImg)

        # fifth part of sequence
        currentRightImgRepeated = self._repeatImageFourTimes(currentRightImg)

        """
        a, s', V(s') are 6 are 11inputs 
        """
        # sixth part of sequence
        actionChunkPlaceholder = self._repeatImageFourTimes(blankPlaceHolder)

        # seventh part of sequence
        futureProprioPlaceholder = self._repeatImageFourTimes(blankPlaceHolder)

        # eighth part of sqeuence
        futureWristImageRepeated = self._repeatImageFourTimes(futureWristImg)

        # ninth part of sequence
        futureLeftImgRepeated = self._repeatImageFourTimes(futureLeftImg)

        # tenth part of sequence
        futureRightImageRepeated = self._repeatImageFourTimes(futureRightImg)

        # 11th part of sequence
        valuePlaceholder = self._repeatImageFourTimes(blankPlaceHolder)
        # B, C, T, H, W
        # dim is 2
        blankPlaceHolderUnsqueeze = rearrange(blankPlaceHolder, "b c h w -> b c 1 h w")
        inputTensor = torch.cat(
            [
                blankPlaceHolderUnsqueeze,
                currentProprioPlaceholder,
                currentWristImgRepeated,
                currentLeftImgRepeated,
                currentRightImgRepeated,
                actionChunkPlaceholder,
                futureProprioPlaceholder,
                futureWristImageRepeated,
                futureLeftImgRepeated,
                futureRightImageRepeated,
                valuePlaceholder,
            ],
            dim=2,
        )

        return inputTensor

    def _VAEForward(self, inputTensor: Tensor):
        return self.vae.forward(inputTensor)

    # rescales between -1 to 1.
    def _normalizePhysical(
        self,
        currentProprio,  # B, 9
        collectedActions,  # B, 32, 7
        futureProprio,  # B,9
    ) -> tuple[Tensor, Tensor, Tensor]:

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

        futurePropScaled = (futureProprio - self.propMin) / (
            (self.propMax - self.propMin) + eps  # pyrefly:ignore
        )
        futurePropNorm = (2.0 * futurePropScaled) - 1.0
        futurePropNorm = torch.clamp(futurePropNorm, min=-1.0, max=1.0)

        return currentPropNorm, actionsNorm, futurePropNorm

    def _latentBlockMaker(self, inputTensor: Tensor) -> Tensor:
        desiredSize = 16 * self.latentSize * self.latentSize
        flattenedTensor = torch.flatten(input=inputTensor, start_dim=1)

        flattenedLength = flattenedTensor.shape[-1]
        repeatedLength = math.ceil(desiredSize / flattenedLength)
        expanded = repeat(
            flattenedTensor, "b r -> b (n r)", r=flattenedLength, n=repeatedLength
        )

        repeatedTensor = expanded[:, :desiredSize]

        reshapedTensor = rearrange(
            repeatedTensor,
            "b (c h w) -> b c h w",
            c=16,
            h=self.latentSize,
            w=self.latentSize,
        )

        return reshapedTensor

    def _rescaleImg(self, img: Tensor) -> Tensor:
        return (img.float() / 127.5) - 1

    @torch.no_grad()
    def forward(
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
    ) -> Tensor:
        # reshapeImages to -1 to 1 before VAE.
        futureValue = rearrange(futureValue, "b -> b 1")
        currentWristImg = self._rescaleImg(currentWristImg)
        currentLeftImg = self._rescaleImg(currentLeftImg)
        currentRightImg = self._rescaleImg(currentRightImg)
        futureWristImg = self._rescaleImg(futureWristImg)
        futureLeftImg = self._rescaleImg(futureLeftImg)
        futureRightImg = self._rescaleImg(futureRightImg)

        # normalize actions, proprio
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

        reshapedValues = self._latentBlockMaker(futureValue)

        # Latent Injection
        # Into index 1, 5,6,10
        # B, C, 11, H, W
        vaeOutput = vaeOutput.clone()
        vaeOutput[:, :, 1, :, :] = reshapedCurrentProp.to(vaeOutput.dtype)
        vaeOutput[:, :, 5, :, :] = reshapedActions.to(vaeOutput.dtype)
        vaeOutput[:, :, 6, :, :] = reshapedFutureProp.to(vaeOutput.dtype)
        vaeOutput[:, :, 10, :, :] = reshapedValues.to(vaeOutput.dtype)

        return vaeOutput
