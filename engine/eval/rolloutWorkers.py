import ray
from ray import serve
from omegaconf import DictConfig, OmegaConf
import h5py
import numpy as np
from PIL import Image
from pathlib import Path
import io
import torch
import gc
from engine.eval.inferenceHelpers import (
    CosmosInferenceSolverEDM,
    CosmosInferenceSolverRecFlow,
    InferenceActionBuilder,
    unNormalizeLatents,
)
from engine.eval.roboCasaEnv import RoboCasaEnvironmentWorker
from models.cosmosLoaderBase import (
    loadCosmosModules,
    EncoderVAE,
    CosmosDiffusionNet,
    ReasonTextEncoder,
)
from models.cosmosLoaderTrained import loadTrainedPolicyModel


@serve.deployment(
    ray_actor_options={"num_cpus": 1, "num_gpus": 1},
    num_replicas=4,
    max_ongoing_requests=64,
)
class CosmosInferencePolicyServer:
    def __init__(self, cfg: DictConfig):
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)  # pyrefly:ignore
        self.device = torch.device("cuda")
        self.cfg = cfg
        self.actionHorizon = cfg.inference.actionHorizon

        self.actionSolverSteps = cfg.inference.solver.actionSteps

        print("Loading Cosmos Model")
        vae, textEncoder, net, config = loadCosmosModules(cfg)

        policyNet = loadTrainedPolicyModel(cfg, net)
        print("Loaded policyNet weights")

        modelVAE = EncoderVAE(vae).to(self.device)
        self.policyNetModel = CosmosDiffusionNet(policyNet).to(self.device)

        self.textEncoder = ReasonTextEncoder(textEncoder)
        self.dynamicTextEmbedCache = {}

        print("All models loaded")
        # load text embeddings onto gpu
        embKeys = np.load(
            Path(cfg.dataset.reasonMMAP, "embeddingkeys.npy"), allow_pickle=True
        )
        embValues = np.load(Path(cfg.dataset.reasonMMAP, "embeddingvalues.npy"))

        self.embIndex = {str(k).strip().lower(): i for i, k in enumerate(embKeys)}
        self.embValues = torch.from_numpy(embValues).float().to(self.device)

        # define solver, action builder, unnormalize latents

        if cfg.model.trainingStyle == "edm":
            self.inferenceSolver = CosmosInferenceSolverEDM(
                cfg=cfg, diffusionModel=self.policyNetModel, device=self.device
            )
        elif cfg.model.trainingStyle == "recflow":
            self.inferenceSolver = CosmosInferenceSolverRecFlow(  # pyrefly:ignore
                cfg=cfg, diffusionModel=self.policyNetModel, device=self.device
            )
        else:
            raise ValueError("incorrect solver in config")

        self.infereceActionBuilder = InferenceActionBuilder(  # pyrefly:ignore
            cfg=cfg, vaeModel=modelVAE, device=self.device
        ).to(self.device)

        self.unNormalizeLatents = unNormalizeLatents(cfg).to(self.device)

    def _getEmbedding(self, textString: str) -> torch.Tensor:
        cleanString = str(textString).strip().lower()

        if cleanString in self.embIndex:
            idx = self.embIndex[cleanString]
            return self.embValues[idx]

        if cleanString in self.dynamicTextEmbedCache:
            return self.dynamicTextEmbedCache[cleanString]

        with torch.no_grad():
            newEmbedding = self.textEncoder(cleanString)
            newEmbedding = newEmbedding.to(self.device).bfloat16()
        self.dynamicTextEmbedCache[cleanString] = newEmbedding

        return newEmbedding

    def _buildActionMask(
        self, vaeInput: torch.Tensor, T: int = 11, H: int = 28, W: int = 28
    ):
        B = vaeInput.shape[0]

        conditionVideoMask = torch.zeros(
            (B, 1, T, H, W), device=self.device, dtype=torch.bfloat16
        )

        conditionVideoMask[:, :, :5, :, :] = 1.0

        return conditionVideoMask

    @serve.batch(max_batch_size=64, batch_wait_timeout_s=0.05)  # pyrefly:ignore
    @torch.no_grad()
    async def predictionActions(
        self, observationDictList: list[dict], taskStringList: list[str]
    ):
        B = len(observationDictList)

        proprioList = []
        leftImgList = []
        rightImgList = []
        wristImgList = []
        textConditionList = []

        for i in range(B):
            obs = observationDictList[i]
            taskString = taskStringList[i]

            proprioList.append(torch.from_numpy(obs["currentProprio"].copy()).float())
            leftImgList.append(
                torch.from_numpy(obs["currentLeftImg"].copy()).permute(2, 0, 1).float()
            )
            rightImgList.append(
                torch.from_numpy(obs["currentRightImg"].copy()).permute(2, 0, 1).float()
            )
            wristImgList.append(
                torch.from_numpy(obs["currentWristImg"].copy()).permute(2, 0, 1).float()
            )

            textConditionList.append(self._getEmbedding(taskString))

        proprioTensor = torch.stack(proprioList).to(self.device)
        leftImgTensor = torch.stack(leftImgList).to(self.device)
        rightImgTensor = torch.stack(rightImgList).to(self.device)
        wristImgTensor = torch.stack(wristImgList).to(self.device)

        textCondition = torch.stack(textConditionList).squeeze(1)

        latentInput = self.infereceActionBuilder(
            currentProprio=proprioTensor,
            currentWristImg=wristImgTensor,
            currentLeftImg=leftImgTensor,
            currentRightImg=rightImgTensor,
        )

        conditioningMask = self._buildActionMask(latentInput)

        denoisedLatent = self.inferenceSolver.runSolver(
            latentInput, textCondition, conditioningMask, self.actionSolverSteps
        )
        extractedData = self.unNormalizeLatents(denoisedLatent)

        physicalActions = extractedData["actions"].cpu().numpy()
        return [
            {
                "actions": physicalActions[i],
                "predictedWristLatent": denoisedLatent[i, :, 7, :, :]
                .cpu()
                .float()
                .numpy(),
                "predictedPrimaryLatent": denoisedLatent[i, :, 8, :, :]
                .cpu()
                .float()
                .numpy(),
                "predictedSecondaryLatent": denoisedLatent[i, :, 9, :, :]
                .cpu()
                .float()
                .numpy(),
            }
            for i in range(B)
        ]


# distributed robocasa wrapper.
@ray.remote(num_cpus=1, num_gpus=0.1)
def distributedRobocasaWorker(
    cfg: DictConfig,
    taskName: str,
    numActionsLength: int,
    maxSteps: int,
    seed: int,
    episodeIDX: int,
    inferenceServer,
    HDF5Writer,
    numScenes: int = 5,
):
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)  # pyrefly:ignore

    roboEnv = RoboCasaEnvironmentWorker(
        cfg=cfg,
        taskName=taskName,
        numActionsLength=numActionsLength,
        seed=seed,
        episodeIDX=episodeIDX,
        numScenes=numScenes,
    )

    observation = roboEnv.reset()
    taskSentence = roboEnv.taskSentence

    done = False
    totalSteps = 0
    while not done and totalSteps < maxSteps:
        result = inferenceServer.predictionActions.remote(
            observation, taskSentence
        ).result()

        actionChunk = result["actions"]

        roboEnv.storeWorldModelLatents(
            wristLatent=result["predictedWristLatent"],
            primaryLatent=result["predictedPrimaryLatent"],
            secondaryLatent=result["predictedSecondaryLatent"],
        )

        stepResult = roboEnv.step(actionChunk)
        observation = stepResult["observation"]
        done = stepResult["done"]
        totalSteps += stepResult["timestep"]

    episodeData = roboEnv.prepareHistoryExport()
    fileName = (
        f"task={taskName}--ep={episodeIDX}--success={episodeData['success']}.hdf5"
    )
    ray.get(HDF5Writer.saveEpisode.remote(episodeData, fileName))
    roboEnv.close()
    return {"success": episodeData["success"], "steps": totalSteps}


@ray.remote(num_cpus=1)
class HDF5Writer:
    def __init__(self, cfg: DictConfig, saveDir: str):
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)  # pyrefly:ignore
        self.cfg = cfg
        self.saveDir = Path(saveDir)
        self.saveDir.mkdir(parents=True, exist_ok=True)

    def _jpegDecode(self, jpegBytes: bytes):
        buf = io.BytesIO(jpegBytes.tobytes())  # pyrefly:ignore
        return np.array(Image.open(buf))

    def _jpegEncode(self, image):
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="JPEG", quality=95)
        return np.frombuffer(buf.getvalue(), dtype=np.uint8)

    def saveEpisode(self, exportDict: dict, fileName: str) -> str:
        savePath = self.saveDir / fileName
        T = exportDict["primary_images"].shape[0]
        dt = h5py.vlen_dtype(np.dtype("uint8"))

        with h5py.File(savePath, "w") as f:
            f.attrs["task_description"] = exportDict["task_description"]
            f.attrs["success"] = bool(exportDict["success"])

            f.create_dataset("actions", data=exportDict["actions"])
            f.create_dataset("proprio", data=exportDict["proprio"])

            for h5Key, dictKey in [
                ("primary_images_jpeg", "primary_images"),
                ("secondary_images_jpeg", "secondary_images"),
                ("wrist_images_jpeg", "wrist_images"),
            ]:
                ds = f.create_dataset(h5Key, shape=(T,), dtype=dt)
                for t in range(T):
                    ds[t] = self._jpegEncode(exportDict[dictKey][t])
            for latentKey in [
                "predicted_wrist_latents",
                "predicted_primary_latents",
                "predicted_secondary_latents",
            ]:
                if latentKey in exportDict:
                    f.create_dataset(latentKey, data=exportDict[latentKey])

        return str(savePath)
