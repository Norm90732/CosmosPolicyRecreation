from omegaconf import DictConfig
import hydra 
import ray 
import webdataset as wds 
from ray.util.queue import Queue 
from typing import List,Dict 
from collections import defaultdict
import glob 
from pathlib import Path
"""
Dataset Uses the Following Concat Order:

Conditioning Subsequence 
(Current Proprio -> Current Wrist Img -> Current Left Img -> Current Right Img)

+

Target Subsequence
(Action Chunk -> Future Proprio -> Future Wrist Img -> Future Right Img -> Future State Value)
"""
#ray this 
def allScenesPreprocess(queue, workerId:int, cfg:DictConfig) -> None:
    outputDir = Path(cfg.dataset.allScenesOutputDir)
    outputDir.mkdir(parents=True,exist_ok=True)
    pattern = str(outputDir/ f"allScenes_workerId_{workerId}_%04d.tar")
    
    with wds.ShardWriter(pattern,maxsize=1e9) as sink: 
        pass 
    
    return None 


#ray this 
def successOnlyPreprocess(queue, workerId:int, cfg:DictConfig) -> None:
    outputDir = Path(cfg.dataset.allSuccessOutputDir)
    outputDir.mkdir(parents=True,exist_ok=True)
    pattern = str(outputDir/ f"allSuccess_workerId_{workerId}_%04d.tar")
    
    with wds.ShardWriter(pattern,maxsize=1e9) as sink: 
        pass 
    return None 
    

#Producer Worker
@ray.remote(num_cpus=1)
def fileProducer(directoryDictionary:dict,queue,numWorkers:int) -> None:
    for taskName, filepaths in directoryDictionary.items():
        for path in filepaths:
            queue.put((taskName, path))
    
    #Ender 
    for _ in range(numWorkers):
        queue.put(None)

    return None 



"""
Parses through downloadDir and extracts Directory Names and HDF5 files 
"""
def fileNameExtractor(cfg:DictConfig,subFolder:str) -> dict[str,list[str]]:
    downloadDir = cfg.dataset.downloadDir
    
    directoryDictionary:dict[str,list[str]] = {}
    
    successOnly = Path(downloadDir) / subFolder
    
    for entry in successOnly.iterdir():
        if entry.is_dir():
            directoryDictionary[entry.name] = [str(p) for p in entry.glob("*/*.hdf5")]
            print(f"Added {entry.name} directory")
        else: 
            print(f"Did not add {entry.name}")
    
    return directoryDictionary


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    allEpisodes = fileNameExtractor(cfg,"all_episodes")
    allSuccesses = fileNameExtractor(cfg,"success_only")
    return None


if __name__ == "__main__":
    main()