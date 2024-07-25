from utils.weights import *
from utils.tester import Tester
from models.plate import LicensePlateModel

def custom_question() -> None:
    inp = input("Do you want to run a custom test? (y/n): ")
    if inp.lower() == 'y':
        image_path = input("Enter the path to the image: ")
        num_digits = 7
        inp = input("Enter the number of digits in the license plate (default 7): ")
        if inp.isdigit():
            num_digits = int(inp)
        custom_test(model, image_path, num_digits)

def custom_test(model, image_path: str, num_digits: int = 7) -> None:
    tester = Tester("./", model)
    tester.num_digits = num_digits
    
    print("Predicting custom image...")

    correct: str = image_path.split("/")[-1].split(".")[0].split("_")[0]
    predicted: str = tester.predict(image_path)

    print(f"True plate: {correct}")
    print(f"Predicted plate: {predicted}")

    print("Custom test finished!")
    exit()

# def custom_test_bimodal(model_number, model_letter, image_path: str, num_digits: int = 7) -> None:
#     tester = Tester("./", model_number, model_letter)
#     tester.num_digits = num_digits
#     
#     print("Predicting custom image...")

#     correct: str = image_path.split("/")[-1].split(".")[0].split("_")[0]
#     predicted: str = tester.predict(image_path)

#     print(f"True plate: {correct}")
#     print(f"Predicted plate: {predicted}")

#     print("Custom test finished!")
#     exit()

if __name__ == "__main__": 
    print("License Plate Tester")

    weightPath: str = ""

    print("Select a model for number and letter recognition...")
    weights: list = weights_list()
    weightPath, ret = weights_select(weights)

    customPreTrained: bool = True
    if ret == -1 or ret == 1:
        print("Using ImageNet pre-trained model...")
        customPreTrained = False
    
    model = LicensePlateModel(customPreTrained=customPreTrained, preTrainedPath=weightPath)

    # weights = weights_list()
    # weightPath, ret = weights_select(weights)

    # customPreTrained = True
    # if ret == -1 or ret == 1:
    #     print("Using ImageNet pre-trained model...")
    #     customPreTrained = False

    # model_letter = LicensePlateModel(customPreTrained=customPreTrained, preTrainedPath=weightPath)
    
    validation_dir = "./assets/images/val/"
    # validation_dir = "/home/lucas/Downloads/placas/"
    tester = Tester(validation_dir, model)

    print("Testing...")
    tester.test()
    print("Testing finished!\t\t\t\t")

    print("Measuring accuracy...")
    tester.accuracy()

    print("Tester finished!")
