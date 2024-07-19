"""Train class to train a model using a given dataset."""

__author__ = "Lucas C. Araujo"
__version__ = "1.0.0"

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

class Train:
    def __init__(self, 
                 model: nn.Module, 
                 criterion: nn.Module, 
                 optimizer: optim.Optimizer,
                 dataLoader: DataLoader,
                 dtype: torch.dtype = torch.float32):
        self.__model: nn.Module = model
        self.__criterion: nn.Module = criterion
        self.__optimizer: optim.Optimizer = optimizer
        self.__dataLoader: DataLoader = dataLoader
        self.__dtype: torch.dtype = dtype
        self.__saveEpochs: bool = False
        self.trainId = torch.randint(0, 1000, (1,)).item()

    @property
    def saveEpochs(self) -> bool:
        return self.__saveEpochs

    @saveEpochs.setter
    def saveEpochs(self, value: bool):
        self.__saveEpochs = value

    def __saveModel(self, filename: str = "model.pth", epoch: int = -1):
        dirpath: str = f"TrainedModels/{self.trainId}/"

        if epoch > 0:
            self.__saveEpochModel(dirpath, filename, epoch)
            return

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        torch.save(self.__model.state_dict(), f"{dirpath}{filename}")

    def __saveEpochModel(self, dirpath: str, filename: str, epoch: int):
        epochDir: str = "epochs/"
        epochDir = f"{dirpath}{epochDir}"

        if not os.path.exists(epochDir):
            os.makedirs(epochDir)

        torch.save(self.__model.state_dict(), 
                        f"{epochDir}{filename}_epoch{epoch}.pth")

    # List all trained models and ask the user to load one
    # If the user wants to load a model, return the path
    # Otherwise, return an empty string
    def __searchPretrained(self) -> str:
        """List all trained models and ask the user to load one.
        
        If the user wants to load a model, return the path.
        Otherwise, return an empty string.
        """
        models: list = os.listdir("TrainedModels/")
        print("Trained models found:")
        for idx, model in enumerate(models):
            print(f"{idx+1}: {model}")
        inp: int = int(input("Which model do you want to load? (0 to skip): "))
        if inp == 0:
            return ""
        return f"TrainedModels/{models[inp-1]}" 

    def __loadPretrained(self, path: str = ""):
        """ Load a pre-trained model from a given path or 
        search for one in the TrainedModels directory.
        
        If the user doesn't want to load a model, just do nothing
        
        If occurs an error loading the model, print it and skip
        """
        inp = input("Do you want to load a pre-trained model? (y/n): ")
        if inp.lower() == 'n':
            return
        elif inp.lower() != 'y':
            print("Invalid input! Skipping...")
            return

        if path == "":
            path = self.__searchPretrained()

        if os.path.exists(path):
            try:
                self.__model.load_state_dict(torch.load(path))
            except Exception as e:
                print(f"Error loading model: {e}")
                return
        
        print("Pre-trained model loaded!")

    def train(self, epochs: int = 10):
        self.__model.type(self.__dtype)

        self.__loadPretrained()

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}", end=' ')
            self.__model.train()
            running_loss = 0.0
            i = 1
            for images, labels in self.__dataLoader:
                print(f"\rEpoch {epoch+1}/{epochs} - Training", end="."*i+" "*(3-i), flush=True)
                i = (i % 3) + 1
                self.__optimizer.zero_grad()
                outputs = self.__model(images)
                loss = self.__criterion(outputs, labels)
                loss.backward()
                self.__optimizer.step()
                running_loss += loss.item()
            if self.saveEpochs:
                self.__saveModel(epoch=epoch+1)
            print(f"\rEpoch {epoch+1}/{epochs} - Training done", flush=True)
            print(f"Loss: {running_loss/len(self.__dataLoader)}")
        
        self.__saveModel("model.pth")
