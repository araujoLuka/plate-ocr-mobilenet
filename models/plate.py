import os
from torch import nn, save, load
from torchvision.models import MobileNetV3, MobileNet_V3_Small_Weights, WeightsEnum
from torchvision.models import mobilenetv3

from utils.trainer import Trainer
weightsDir: str = Trainer.weightsDir

class LicensePlateModel(MobileNetV3):
    def __init__(self, 
                 numClasses:int = 36, 
                 customPreTrained: bool = False,
                 preTrainedPath: str = ""):
        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf("mobilenet_v3_small")
        weights: WeightsEnum = MobileNet_V3_Small_Weights.verify("DEFAULT")
        super(LicensePlateModel, self).__init__(
                inverted_residual_setting, last_channel, num_classes=len(weights.meta["categories"]))
        self.load_state_dict(weights.get_state_dict(check_hash=True))
        self.classifier[3] = nn.Linear(self.classifier[3].in_features, numClasses)
        if customPreTrained:
            try:
                self.load(preTrainedPath)
            except Exception as e:
                print(f"Error loading pre-trained model: {e}")
                print("Training from scratch...")
     
    def save(self, filename: str = "model", epoch: int = -1, dirpath: str = "") -> str:
        """Save the model to a given path. 
        Return the path where the model was saved."""
        if dirpath == "":
            dirpath = weightsDir
        
        if epoch > 0:
            return self.__saveByEpoch(filename, epoch, dirpath)
        
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
          
        if not filename.endswith(".pth"):
            filename += ".pth"
        modelPath: str = f"{dirpath}{filename}"
        save(self.state_dict(), modelPath)
        
        return modelPath
    
    def load(self, path: str = ""):
        """ Load a pre-trained model from a given path or 
        search for one in the 'weightsDir' path.
        
        If the user doesn't want to load a model, just do nothing
        
        If occurs an error loading the model, print it and skip
        """
        if path == "":
            path = self.__searchPretrained()
        
        if os.path.exists(path):
            try:
                print("Loading pre-trained model...")
                print(f"Loading model: {path}")
                self.load_state_dict(load(path))
            except Exception as e:
                print(f"Error loading model: {e}")
                return
        
        print("Pre-trained model loaded!")
    
    def __saveByEpoch(self, filename: str, epoch: int, dirpath: str) -> str:
        """Save the model to a given path with the epoch number. 
        Return the path where the model was saved."""
        epochDir: str = "epochs/"
        epochDir = f"{dirpath}{epochDir}"

        if not os.path.exists(epochDir):
            os.makedirs(epochDir)

        epochToken: str = "epoch" + str(epoch).zfill(2)
        
        if filename.endswith(".pth"):
            filename.strip(".pth")
        modelPath: str = f"{epochDir}{filename}_{epochToken}.pth"
        save(self.state_dict(), modelPath)
        
        return modelPath

    def __searchPretrained(self) -> str:
        """List all trained models and ask the user to load one.
        
        If the user wants to load a model, return the path.
        Otherwise, return an empty string.
        """
        # Make a list with all main trained models for each training ID in 'weights'
        trainings: list = os.listdir(weightsDir)
        models: list = []
        for training in trainings:
            for model in os.listdir(f"{weightsDir}{training}"):
                if model.endswith(".pth"):
                    models.append(f"{training}/{model}")
        
        print("Trained models found:")
        for idx, model in enumerate(models):
            print(f"{idx+1}: {model}")
        
        inp: int = int(input("Which model do you want to load? (0 to skip): "))
        if inp == 0:
            raise Exception("User skipped loading a pre-trained model")
        if inp < 1 or inp > len(models):
            raise Exception("Invalid input! Skipping...")
        
        print(f"Loading model: {models[inp-1]}")
        return f"{weightsDir}{models[inp-1]}"
