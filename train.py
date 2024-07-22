import argparse
from torch import nn, optim

from utils.trainer import Trainer
from data.dataset import LicensePlateDataset
from data.dataloader import LicensePlateDataLoader
from models.plate import LicensePlateModel

if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="Train a model for license plate recognition.")
    argParser.add_argument("-l", "--letter", action="store_true", help="Train a model only for letter recognition.")
    argParser.add_argument("-n", "--number", action="store_true", help="Train a model only for number recognition.")
    argParser.add_argument("--pre-trained", type=str, help="Path to a pre-trained model.", default="")
    argParser.add_argument("-e", "--epochs", type=int, help="Number of epochs to train the model.", default=10)
    argParser.add_argument("--skip-question", action="store_true", help="Skip the question to load a pre-trained model.", default=False)
    args = argParser.parse_args()

    trainDataPath = "./assets/images/train/"
    if args.letter:
        trainDataPath = "./assets/images/train/letters/"
    elif args.number:
        trainDataPath = "./assets/images/train/numbers/"

    if args.letter and args.number:
        print("Invalid arguments! Please choose only one model to train.")
        exit()

    dataset = LicensePlateDataset(trainDataPath)
    dataloader = LicensePlateDataLoader(dataset)

    preTrainedPath = args.pre_trained
    usePretrained = False

    if preTrainedPath == "" and not args.skip_question:
        inp = input("Do you want to load a pre-trained model? (y/n): ")
        if inp.lower() == 'y':
            usePretrained = True
        elif inp.lower() != 'n':
            print("Invalid input! Skipping...")
    
    if preTrainedPath != "":
        usePretrained = True

    model = LicensePlateModel(customPreTrained=usePretrained, preTrainedPath=preTrainedPath)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    trainer = Trainer(model, criterion, optimizer, dataloader)
    modelSaveName = "model"
    trainer.saveEpochs = True
    if args.number:
        trainer.trainId = trainer.trainId.replace("TRAIN", "NUMBER_TRAIN")
        modelSaveName = "model_number"
    elif args.letter:
        trainer.trainId = trainer.trainId.replace("TRAIN", "LETTER_TRAIN")
        modelSaveName = "model_letter"
    print()

    print("Training... (TRAINING_ID: " + trainer.trainId + ")")
    trainer.train(epochs=args.epochs)
    print("Training finished!")

    print("Model saved at:", model.save(modelSaveName))
