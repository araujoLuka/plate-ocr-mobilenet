import os
import torch
from PIL import Image

from assets.metadata import DIGITS_CATEGORIES
from models.plate import LicensePlateModel
from utils.digit import DigitExtractor
from utils.transforms import plateModelTransform

class Tester:
    def __init__(self, data_dir: str, model_number: LicensePlateModel, model_letter: LicensePlateModel):
        self.__data_dir: str = data_dir
        self.__model_number: torch.nn.Module = model_number
        self.__model_letter: torch.nn.Module = model_letter
        self.__total: int = 0
        self.__correct: int = 0
        self.__errors: list = []
        self.__classes: list = DIGITS_CATEGORIES
        self.__idx_to_class: dict = { idx: cls for idx, cls in enumerate(self.__classes) }
        self.num_digits: int = 7
        self.model_eval()

    def model_eval(self) -> None:
        self.__model_number.eval()
        self.__model_letter.eval()

    def __gen_output(self, image: torch.Tensor, i: int) -> torch.Tensor:
        if i < 3:
            return self.__model_letter(image)
        return self.__model_number(image)

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
        
        for i, digit_image in enumerate(segmented_digits):
            digit_image = digit_image.unsqueeze(0)
            with torch.no_grad():
                output = self.__gen_output(digit_image, i)
            _, max_idxs = torch.max(output.data, 1)
            predicted: int = int(max_idxs.item())
            predicted_char: str = self.__idx_to_class[predicted]
            predicted_chars.append(predicted_char)
        
        predicted_plate: str = "".join(predicted_chars[:3]) + "-" + "".join(predicted_chars[3:])
        print(f"\rPredicted plate: {predicted_plate}", end=" ")
        return predicted_plate
    
    def accuracy(self) -> float:
        if self.__errors:
            print("Error list:")
            for true_plate, predicted_plate in self.__errors:
                print(f"True plate: {true_plate}, Predicted plate: {predicted_plate}")
            print("--- Error list end ---")
        
        print(f"Total  images: {self.__total}")
        print(f"Total correct: {self.__correct}")
        acc: float = self.__correct / self.__total * 100
        print(f"Accuracy: {self.__correct}/{self.__total} ({acc:.2f}%)")
        return acc

