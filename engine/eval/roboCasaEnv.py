import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import robosuite
import robocasa  #pyrefly:ignore 

class RoboCasaEnvironmentWorker():
    def __init__(self,taskName:str,numActionsLength:int,seed:int,episodeIDX:int,numScenes:int=5) -> None:
        sceneIndex = (episodeIDX //10) % numScenes

        layoutStyleIds = [(1,1), (2,2), (4,4), (6,9), (7,10)]
        
        layoutStyle = layoutStyleIds[sceneIndex]
        
        env_kwargs = {
            "env_name": taskName,
            "robots": "PandaMobile",
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "camera_names": [
            "robot0_agentview_left",     
            "robot0_agentview_right",   
            "robot0_eye_in_hand"        #wrist
        ],
            "camera_widths": 224,             
            "camera_heights": 224,
            "seed": seed,
            "layout_and_style_ids": (layoutStyle,),
        }
    
        self.env = robosuite.make(**env_kwargs)
        
        self.numActions = numActionsLength #number of steps to execute 
        
    def _extractAndFormat(self,observationDictionary:dict):
        leftImg = observationDictionary["robot0_agentview_left_image"][::-1]
        rightImg = observationDictionary["robot0_agentview_right_image"][::-1]
        wristImg = observationDictionary["robot0_eye_in_hand_image"][::-1]
        
        gripperPos = observationDictionary["robot0_gripper_qpos"]
        eefPos = observationDictionary["robot0_eef_pos"]
        eefQuat = observationDictionary["robot0_eef_quat"]
        
        proprio = np.concatenate([
            gripperPos,eefPos,eefQuat
        ])
        
        return {
            "currentProprio":proprio,
            "currentLeftImg":leftImg,
            "currentRightImg": rightImg,
            "currentWristImg": wristImg
        }

    def reset(self)->dict:
        rawObservation = self.env.reset()
        #policy code uses stabilizes the env 
        for _ in range(10):
            rawObservation, _, _, _ = self.env.step(np.zeros(self.env.action_dim))
            
        return self._extractAndFormat(rawObservation)
    
    
    def step(self,actionSequence): 
        #predict 32, 7 chunk -> 16, 12 
        if actionSequence.shape[-1] == 7:
            seqLen = actionSequence.shape[0]
            mobileBase = np.zeros((seqLen, 5))
            mobileBase[:, -1] = -1.0 
            actionSequence = np.concatenate([actionSequence, mobileBase], axis=1)
            
        numSteps = min(self.numActions,actionSequence.shape[0])
        
        for i in range(0,numSteps):
            predictedAction = actionSequence[i]
            nextObservation, reward, done, info = self.env.step(predictedAction)
            
            success = self.env._check_success()
            if success == True:
                break 
        
        return {
            "success": bool(success),
            "timestep": i + 1,
            "observation": self._extractAndFormat(nextObservation),
            "reward": reward,
            "info": info,
            "done": done or success
        }

            
    def close(self):
        self.env.close()    
        del self.env 
    

