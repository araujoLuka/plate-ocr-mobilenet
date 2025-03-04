"""Train class to train a model using a given dataloader."""

__author__ = "Lucas C. Araujo"
__version__ = "1.0.0"

import os
import torch
import time
from torch import nn, optim
from torch.utils.data import DataLoader

from utils.weights import weightsDir

class Trainer:
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
        self.__trainDir = ""
        self.__trainId: str = self.generateTrainId()

    @property
    def trainId(self) -> str:
        return self.__trainId

    @trainId.setter
    def trainId(self, value: str):
        self.__trainId = value
        self.__trainDir = f"{weightsDir}{value}/"
        if os.path.exists(self.__trainDir):
            print(f"\rTrain ID {value} already exists! Creating another...", end=" ")
            self.trainId = value[:-3] + str(int(value[-3:])+1).zfill(3)

    def generateTrainId(self) -> str:
        i: int = 1
        date: str = time.strftime("%Y%m%d")
        id: str = f"{date}_TRAIN_{str(i).zfill(3)}"
        path: str = f"{weightsDir}{id}/"
        while os.path.exists(path):
            i += 1
            id = f"{date}_TRAIN_{str(i).zfill(3)}"
            path = f"{weightsDir}{id}/"
        self.__trainDir = path
        return id
    
    @property
    def saveEpochs(self) -> bool:
        return self.__saveEpochs

    @saveEpochs.setter
    def saveEpochs(self, value: bool):
        self.__saveEpochs = value
     
    @property
    def trainDir(self) -> str:
        return self.__trainDir

    @trainDir.setter
    def trainDir(self, value: str):
        self.__trainDir = value
    
    def train(self, epochs: int = 10):
        self.__model.type(self.__dtype)

        epochWithLessLoss: int = 0
        bestEpochPath: str = ""
        epochPath: str = ""
        lessLoss: float = float("inf")
        for epoch in range(epochs):
            epochStr = str(epoch+1).zfill(len(str(epochs)))
            print(f"Epoch {epochStr}/{epochs}", end=' ')
            self.__model.train()
            running_loss = 0.0
            i = 1
            for images, labels in self.__dataLoader:
                dotEnd = "."*i + " "*(3-i) + " "
                print(f"\rEpoch {epochStr}/{epochs} - Training", end=dotEnd, flush=True)
                i = (i % 3) + 1
                self.__optimizer.zero_grad()
                outputs = self.__model(images)
                loss = self.__criterion(outputs, labels)
                loss.backward()
                self.__optimizer.step()
                running_loss += loss.item()

            if self.saveEpochs:
                epochPath = self.__model.save(epoch=epoch+1, dirpath=self.__trainDir)

            print(f"\rEpoch {epochStr}/{epochs} - Training done\t\t\t\t\t", flush=True)
            print(f"Loss: {running_loss/len(self.__dataLoader)}")
            lr = self.__optimizer.param_groups[0]['lr']
            print(f"Learning rate: {lr:.6f}")
            lr *= 0.9
            lr = float(f"{lr:.6f}") # Truncate to 6 decimal
            lr = max(lr, 0.00001)
            for param_group in self.__optimizer.param_groups:
                param_group['lr'] = lr
            print(f"Rate updated: {lr:.6f}")

            if running_loss < lessLoss:
                lessLoss = running_loss
                epochWithLessLoss = epoch
                bestEpochPath = epochPath

            if running_loss/len(self.__dataLoader) < 0.001:
                print("Loss is too low! Stopping training...")
                break
        
        print(f"Training finished! Best epoch: {epochWithLessLoss+1}")
        print("Saving it...")
        self.__model.load(bestEpochPath)
        self.__model.save("model.pth", dirpath=self.__trainDir)
