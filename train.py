from torch import nn, optim

from utils.trainer import Trainer
from data.dataset import LicensePlateDataset
from data.dataloader import LicensePlateDataLoader
from models.plate import LicensePlateModel

if __name__ == "__main__":
    dataset = LicensePlateDataset("./temp/train/")
    dataloader = LicensePlateDataLoader(dataset)

    inp = input("Do you want to load a pre-trained model? (y/n): ")
    usePretrained = False
    if inp.lower() == 'y':
        usePretrained = True
    elif inp.lower() != 'n':
        print("Invalid input! Skipping...")

    preTrainedPath = ""
    if usePretrained:
        inp = input("Do you want to load a specific model? (y/n): ")
        if inp.lower() == 'y':
            preTrainedPath = input("Enter the path to the model: ")

    model = LicensePlateModel(customPreTrained=usePretrained, preTrainedPath=preTrainedPath)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, criterion, optimizer, dataloader)
    trainer.saveEpochs = True

    print("Training... (TRAINING_ID:", trainer.trainId, ")")
    trainer.train(epochs=10)
    print("Training finished!")

    print("Model saved at:", model.save("model"))
