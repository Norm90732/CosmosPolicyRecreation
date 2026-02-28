from omegaconf import DictConfig
import hydra
import ray
import webdataset as wds
from ray.util.queue import Queue
from typing import List, Dict
from collections import defaultdict
import glob
from pathlib import Path
import numpy as np
from PIL import Image 
import io 
"""
Dataset Uses the Following Concat Order:

Conditioning Subsequence 
(Current Proprio -> Current Wrist Img -> Current Left Img -> Current Right Img)

+

Target Subsequence
(Action Chunk -> Future Proprio -> Future Wrist Img -> Future Right Img -> Future State Value)
"""


@ray.remote(num_cpus=1)
def successOnlyPreprocess(queue, workerId: int, cfg: DictConfig) -> None:
    import h5py as h5

    # Define output webdataset save directory
    outputDir = Path(cfg.dataset.allSuccessOutputDir)
    outputDir.mkdir(parents=True, exist_ok=True)
    pattern = str(outputDir / f"allSuccess_workerId_{workerId}_%04d.tar")
    actionDim = 7  # The full RoboCasa action space has 12 dims, but we only use the first 7 (the rest are for the mobile base)

    # begin shard writer with data brought from queue.
    with wds.ShardWriter(pattern, maxsize=1e9) as sink:
        while True:
            message = queue.get()
            # Breaks Loop and ends Worker
            if message is None:
                print(f"Worker {workerId} finished.")
                break

            # Receives Data from Queue
            taskName, path = message

            try:
                # Read hdf5 file
                with h5.File(path, "r") as f:
                    # read demo groups
                    demoGroups = f["data"].keys()
                    sortedDemoGroups = sorted(
                        demoGroups, key=lambda x: int(x.split("_")[1])
                    )

                    for demoIdx in sortedDemoGroups:
                        demo = f[f"data/{demoIdx}"]
                        taskDescription = demo.attrs["task_description"]

                        isJpeg = (
                            "robot0_agentview_left_rgb_jpeg" in f[f"data/{demoIdx}/obs"]
                        )

                        if isJpeg is True:
                            numSteps = len(
                                f[f"data/{demoIdx}/obs/robot0_agentview_left_rgb_jpeg"]
                            )
                        else:
                            numSteps = len(
                                f[f"data/{demoIdx}/obs/robot0_agentview_left_rgb"]
                            )
                        monteReturnsArray= monteCarloRewards(numSteps=numSteps,
                                                             discountFactor=cfg.dataset.monteCarloRewards.discountFactor,
                                                             terminalReward=cfg.dataset.monteCarloRewards.terminalReward)
                        for tIdx in range(0, numSteps):
                            futureTidx = min(tIdx + 32, numSteps - 1)
                            futureValue = monteReturnsArray[futureTidx]
                            
                            # action state collection
                            collectedActions = demo["actions"][
                                tIdx:futureTidx, :actionDim
                            ].astype(np.float32)
                            
                            
                            #padding and repeat 
                            
                            if len(collectedActions) < 32:
                                diff = 32 - len(collectedActions)
                                repeatedAction = collectedActions[-1]
                                toAppend = np.full((diff, actionDim), repeatedAction)
                                collectedActions = np.concatenate([collectedActions,toAppend])
                                

                            currentProprio = demo["robot_states"][tIdx].astype(
                                np.float32
                            )
                            futureProprio = demo["robot_states"][futureTidx].astype(
                                np.float32
                            )

                            # image saving (stacking done in training to save storage)

                            if isJpeg is True:
                                # T, H, W, 3 uint8
                                currentWristImg = demo["obs"]["robot0_eye_in_hand_rgb_jpeg"][tIdx]  # saved as bytes
                                futureWristImg = demo["obs"]["robot0_eye_in_hand_rgb_jpeg"][futureTidx]

                                currentLeftImg = demo["obs"]["robot0_agentview_left_rgb_jpeg"][tIdx]
                                futureLeftImg = demo["obs"]["robot0_agentview_left_rgb_jpeg"][futureTidx]

                                currentRightImg = demo["obs"]["robot0_agentview_right_rgb_jpeg"][tIdx]
                                futureRightImg = demo["obs"]["robot0_agentview_right_rgb_jpeg"][futureTidx]
                            else:
                                currentWristImg = demo["obs"]["robot0_eye_in_hand_rgb"][tIdx]
                                currentWristImg = numpyToBytes(currentWristImg)
                                
                                futureWristImg = demo["obs"]["robot0_eye_in_hand_rgb"][futureTidx]
                                futureWristImg = numpyToBytes(futureWristImg)
                                
                                currentLeftImg = demo["obs"]["robot0_agentview_left_rgb"][tIdx]
                                currentLeftImg = numpyToBytes(currentLeftImg)
                                
                                futureLeftImg = demo["obs"]["robot0_agentview_left_rgb"][futureTidx]
                                futureLeftImg = numpyToBytes(futureLeftImg)
                                
                                currentRightImg = demo["obs"]["robot0_agentview_right_rgb"][tIdx]
                                currentRightImg = numpyToBytes(currentRightImg)
                                
                                futureRightImg = demo["obs"]["robot0_agentview_right_rgb"][futureTidx]
                                futureRightImg = numpyToBytes(futureRightImg)
                            
                            #saving to format 
                            fileNameClean = Path(path).stem 
                            sample = {
                                "__key__": f"{taskName}_{fileNameClean}_{demoIdx}_{tIdx:05d}",
                                "futureValue.npy": futureValue,
                                "collectedActions.npy":collectedActions,
                                 "currentProprio.npy":currentProprio,
                                 "futureProprio.npy":futureProprio,
                                 "currentWristImg.jpg":currentWristImg,
                                 "futureWristImg.jpg": futureWristImg,
                                 "currentLeftImg.jpg":currentLeftImg,
                                 "futureLeftImg.jpg":futureLeftImg,
                                 "currentRightImg.jpg":currentRightImg,
                                 "futureRightImg.jpg":futureRightImg,
                                 "taskDescription.txt": taskDescription, 
                                 "isSuccess.json": True
                            }
                            sink.write(sample)

            except Exception as e:
                print(f"Error on {path}: {e}")

    return None

# ray this
def allScenesPreprocess(queue, workerId: int, cfg: DictConfig) -> None:
    outputDir = Path(cfg.dataset.allScenesOutputDir)
    outputDir.mkdir(parents=True, exist_ok=True)
    pattern = str(outputDir / f"allScenes_workerId_{workerId}_%04d.tar")

    with wds.ShardWriter(pattern, maxsize=1e9) as sink:
        pass

    return None


# Producer Worker
@ray.remote(num_cpus=1)
def fileProducer(directoryDictionary: dict, queue, numWorkers: int) -> None:
    for taskName, filepaths in directoryDictionary.items():
        for path in filepaths:
            queue.put((taskName, path))

    # Ender
    for _ in range(numWorkers):
        queue.put(None)

    return None





"""
Parses through downloadDir and extracts Directory Names and HDF5 files 
"""
def fileNameExtractor(cfg: DictConfig, subFolder: str) -> dict[str, list[str]]:
    downloadDir = cfg.dataset.downloadDir

    directoryDictionary: dict[str, list[str]] = {}

    successOnly = Path(downloadDir) / subFolder

    for entry in successOnly.iterdir():
        if entry.is_dir():
            directoryDictionary[entry.name] = [str(p) for p in entry.glob("*/*.hdf5")]
            print(f"Added {entry.name} directory")
        else:
            print(f"Did not add {entry.name}")

    return directoryDictionary


"""
Monte-Carlo Rewards Function

MDP is <S, A, T, R, H> 
S = set of states,
A = set of actions, 
T = state transition function 
R = reward function 
H = time horizon 


Rt = {
    0 t< T-1 ; 
    t = T -1 ; 
}
"""

def monteCarloRewards(numSteps: int, discountFactor: float, terminalReward: float):
    if terminalReward == 0:
        return np.full(numSteps, -1.0, dtype=np.float32)
    t = np.arange(numSteps)
    Gt = (discountFactor ** (numSteps - 1 - t)) * terminalReward
    Gt = 2 * Gt / terminalReward - 1
    
    return Gt #returns array 


def numpyToBytes(array: np.ndarray) -> bytes:
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG",quality=95)
    return buffer.getvalue()

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    allEpisodes = fileNameExtractor(cfg, "all_episodes")
    allSuccesses = fileNameExtractor(cfg, "success_only")
    return None


if __name__ == "__main__":
    main()
