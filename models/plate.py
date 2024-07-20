"""Model class to solve the License Plate classification problem."""

__author__ = "Lucas C. Araujo"

import os
from torch import nn, save, load
from torchvision.models import MobileNetV3, MobileNet_V3_Small_Weights, WeightsEnum
from torchvision.models import mobilenetv3

from utils.weights import *

class LicensePlateModel(MobileNetV3):
    def __init__(
            self, 
            numClasses:int = 36, 
            customPreTrained: bool = False,
            preTrainedPath: str = ""):
        """
        License Plate Model class, based on the MobileNetV3 architecture.
        
        Initialize the LicensePlateModel class with the given number of classes.
        The model uses the 'mobilenet_v3_small' configuration. 
        The model is pre-trained with the IMAGENET dataset.
        
        Optionally, the user can load a pre-trained model from the 'weightsDir' path or
        from a given path.
        
        if 'customPreTrained' is True and 'preTrainedPath' is empty, it searches for a pre-trained
        and shows a list of models to the user. The user can choose one to load.
         
        Args:
            numClasses (int, optional): Number of classes for the model. Defaults to 36.
            customPreTrained (bool, optional): Load a pre-trained model. Defaults to False.
            preTrainedPath (str, optional): Path to the pre-trained model. Defaults to "".
        """

        # Load the MobileNetV3 model with the 'mobilenet_v3_small' configuration
        # 1. Get mobilenet_v3_small configuration
        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf("mobilenet_v3_small")
        # 2. Get the IMAGENET weights
        weights: WeightsEnum = MobileNet_V3_Small_Weights.verify("DEFAULT")
        # 3. Initialize the model with the configuration and the default number of classes
        super(LicensePlateModel, self).__init__(
                inverted_residual_setting, last_channel, num_classes=len(weights.meta["categories"]))
        # 4. Load the IMAGENET weights into the model
        self.load_state_dict(weights.get_state_dict(check_hash=True))

        # After loaded the model, change the last layer to match the number of classes
        self.classifier[3] = nn.Linear(self.classifier[3].in_features, numClasses)

        # Load a pre-trained model if the user wants
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
        weights: list = weights_list()
        path: str = ""
        ret: int = 0
        path, ret = weights_select(weights)

        if ret == 1:
            raise Exception("User skipped loading a pre-trained model")
        
        if ret == -1:
            raise Exception("Invalid input! Skipping...")
        
        return path
