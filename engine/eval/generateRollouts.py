import os 
os.environ["JAXTYPING_DISABLE"] = "1"
from engine.eval.rolloutWorkers import CosmosInferencePolicyServer,HDF5Writer,distributedRobocasaWorker
from omegaconf import DictConfig, OmegaConf
import ray 
import json 
from pathlib import Path 
from ray import serve 

def runRollOutCollection(cfg:DictConfig):
    ray.init()
    
    serve.start() 
    
    cfgDict = OmegaConf.to_container(cfg, resolve=True)
    app = CosmosInferencePolicyServer.bind(cfgDict) #pyrefly:ignore 
    inferenceServer = serve.run(app)
    HDF5WriterObj = HDF5Writer.remote(cfgDict, cfg.inference.rolloutSaveDir) #pyrefly:ignore 
    jsonResultsSaveDir = cfg.inference.jsonSaveDir
    
    numTrials = cfg.inference.numTrialsPerTask
    numScenes = cfg.inference.numScenes
    openLoopSteps = cfg.inference.openLoopSteps
    taskMaxSteps = OmegaConf.to_container(cfg.inference.taskMaxSteps)
    
    
    
    futures = []
    for baseSeed in cfg.inference.baseSeeds:
        for taskName in cfg.inference.robocasaEnvs:
            maxSteps = taskMaxSteps[taskName]  #pyrefly:ignore 
            for episodeIDX in range(numTrials):
                envSeed = baseSeed * episodeIDX * 256
                future = distributedRobocasaWorker.remote(
                    cfg=cfgDict, #pyrefly:ignore 
                    taskName=taskName,  #pyrefly:ignore 
                    numActionsLength=openLoopSteps,  #pyrefly:ignore 
                    seed=envSeed,  #pyrefly:ignore 
                    episodeIDX=episodeIDX,  #pyrefly:ignore 
                    inferenceServer=inferenceServer,  #pyrefly:ignore 
                    HDF5Writer=HDF5WriterObj,  #pyrefly:ignore 
                    maxSteps=maxSteps,  #pyrefly:ignore 
                    numScenes=numScenes,  #pyrefly:ignore 
                )
                futures.append((baseSeed, taskName, episodeIDX, future))

    episodeLogs = []
    taskSuccesses = {task: [] for task in cfg.inference.robocasaEnvs}
    
    for baseSeed, taskName, episodeIDX, future in futures:
        result = ray.get(future)
        taskSuccesses[taskName].append(result["success"])
        episodeLogs.append({
            "seed": baseSeed,
            "task": taskName,
            "episodeIDX": episodeIDX,
            "success": result["success"],
            "steps": result["steps"],
        })

    # Build summary
    taskSummary = {
        taskName: {
            "successes": sum(successes),
            "total": len(successes),
            "successRate": round(sum(successes) / len(successes) * 100, 2)
        }
        for taskName, successes in taskSuccesses.items()
    }
    allRates = [v["successRate"] for v in taskSummary.values()]
    
    fullResults = {
        "averageSuccessRate": round(sum(allRates) / len(allRates), 2),
        "taskSummary": taskSummary,
        "episodeLogs": episodeLogs,
    }

    savePath = Path(cfg.inference.jsonSaveDir) / "rollOutResults.json"
    savePath.parent.mkdir(parents=True, exist_ok=True)
    with open(savePath, "w") as f:
        json.dump(fullResults, f, indent=2)
    
    print(f"Results saved to {savePath}")
    ray.shutdown()

if __name__ == "__main__":
    cfg = OmegaConf.load("configs/config.yaml")
    cfg.dataset = OmegaConf.load("configs/dataset/robocasaRollout.yaml")
    cfg.model = OmegaConf.load("configs/model/policy.yaml")
    cfg.inference = OmegaConf.load("configs/inference/rollout.yaml")

    runRollOutCollection(cfg) #pyrefly:ignore 
    