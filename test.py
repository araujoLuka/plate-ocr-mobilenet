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

if __name__ == "__main__": 
    print("License Plate Tester")

    customPreTrained: bool = True
    weightPath: str = ""
    weights: list = weights_list()
    print(f"{len(weights)+1}: Custom model")
    weights.append("Custom model")

    weightPath, ret = weights_select(weights)

    if ret == -1 or ret == 1:
        print("Using ImageNet pre-trained model...")
        customPreTrained = False
    
    model = LicensePlateModel(customPreTrained=True, preTrainedPath=weightPath)
    
    # custom_question()
    
    validation_dir = "./assets/images/val/"
    tester = Tester(validation_dir, model)

    print("Testing...")
    tester.test()
    tester.accuracy()

    print("Tester finished!")
