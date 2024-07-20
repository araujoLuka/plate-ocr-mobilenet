import os
import torch
from PIL import Image

from assets.metadata import DIGITS_CATEGORIES
from utils.digit import DigitExtractor
from utils.transforms import plateModelTransform
from utils.weights import *
from models.plate import LicensePlateModel

class Tester:
    def __init__(self, data_dir: str, model: torch.nn.Module):
        self.__data_dir: str = data_dir
        self.__model: torch.nn.Module = model
        self.__total: int = 0
        self.__correct: int = 0
        self.__errors: list = []
        self.__classes: list = DIGITS_CATEGORIES
        self.__idx_to_class: dict = { idx: cls for idx, cls in enumerate(self.__classes) }
        self.num_digits: int = 7
        self.__model.eval()
    
    def __transform(self, image: Image.Image) -> Image.Image:
        return plateModelTransform(image)

    def segmentation(self, image_path: str) -> list[Image.Image]:
        digits: list = []
        for i in range(self.num_digits):
            dgExtractor: DigitExtractor = DigitExtractor(
                    image_path, 
                    i+1, 
                    plate_length=self.num_digits)
            digit_image: Image.Image = dgExtractor.digit
            digit_image = self.__transform(digit_image)
            digits.append(digit_image)
        return digits
    
    def test(self) -> None:
        for file in os.listdir(self.__data_dir):
            if file.endswith(".jpg"):
                self.__total += 1
                true_plate: str = file.split(".")[0]
                predicted_plate: str = self.predict(f"{self.__data_dir}/{file}")
                if true_plate == predicted_plate:
                    self.__correct += 1
                else:
                    self.__errors.append((true_plate, predicted_plate))
    
    def predict(self, image_path: str) -> str:
        segmented_digits: list = self.segmentation(image_path)
        predicted_chars: list = []
        
        for digit_image in segmented_digits:
            digit_image = digit_image.unsqueeze(0)
            with torch.no_grad():
                output = self.__model(digit_image)
            _, max_idxs = torch.max(output.data, 1)
            predicted: int = int(max_idxs.item())
            predicted_char: str = self.__idx_to_class[predicted]
            predicted_chars.append(predicted_char)
        
        predicted_plate: str = "".join(predicted_chars[:3]) + "-" + "".join(predicted_chars[3:])
        print(f"\rPredicted plate: {predicted_plate}", end=" ")
        return predicted_plate
    
    def accuracy(self) -> float:
        if self.__errors:
            print("Erros:")
            for true_plate, predicted_plate in self.__errors:
                print(f"Placa verdadeira: {true_plate}, Placa inferida: {predicted_plate}")
            print("--- Fim da lista de erros ---")
        
        print(f"\rTotal de imagens: {self.__total}")
        print(f"Total de acertos: {self.__correct}")
        acc: float = self.__correct / self.__total * 100
        print(f"Acurácia: {self.__correct}/{self.__total} ({acc:.2f}%)")
        return acc

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
