import os
import torch
from PIL import Image

from assets.metadata import IDX_TO_CLASS
from models.plate import LicensePlateModel
from utils.image import ImageProcessor
from utils.transforms import baseTransform

class Tester:
    def __init__(self, data_dir: str, model: LicensePlateModel):
        self.__data_dir: str = data_dir
        self.__model: LicensePlateModel = model
        self.__total: int = 0
        self.__correct: int = 0
        self.__errors: list = []
        self.__idx_to_class: dict = IDX_TO_CLASS
        self.num_digits: int = 7
        self.__model.eval()

    def test(self) -> None:
        for file in os.listdir(self.__data_dir):
            if file.endswith(".jpg"):
                self.__total += 1
                true_plate: str = file.split(".")[0]
                predicted_plate: str = self.predict(f"{self.__data_dir}{file}")
                if true_plate == predicted_plate:
                    self.__correct += 1
                else:
                    self.__errors.append((true_plate, "Prediction error: " + predicted_plate))
    
    def predict(self, image_path: str) -> str:
        segmented_digits: list = self.__segment(image_path)
        predicted_chars: list = []

        true_plate: str = image_path.split("/")[-1].split(".")[0]
        if len(segmented_digits) != len(true_plate) - 1:
            print(f"\rError: {true_plate} segmentation has {len(segmented_digits)} elements while true plate has {len(true_plate) - 1} elements.")
            self.__errors.append((true_plate, f"Segmentation error: {len(segmented_digits)} elements (correct: {len(true_plate) - 1})"))

        for i, digit_image in enumerate(segmented_digits):
            digit_image = digit_image.unsqueeze(0)
            if self.__model.on_gpu:
                digit_image = digit_image.cuda()
            with torch.no_grad():
                output = self.__gen_output(digit_image, i)
            _, max_idxs = torch.max(output.data, 1)
            predicted: int = int(max_idxs.item())
            predicted_char: str = self.__idx_to_class[predicted]
            predicted_chars.append(predicted_char)
        
        predicted_plate: str = "".join(predicted_chars[:3] + ["-"] + predicted_chars[3:])
        print(f"\rPredicted plate: {predicted_plate}", end="        ")
        return predicted_plate
    
    def accuracy(self) -> float:
        if self.__errors:
            print("Error list:")
            for plate, error in self.__errors:
                print(f"Plate: {plate}, Error: {error}")
            print("--- Error list end ---")
        
        print(f"Total  images: {self.__total}")
        print(f"Total correct: {self.__correct}")
        acc: float = self.__correct / self.__total * 100
        print(f"Accuracy: {self.__correct}/{self.__total} ({acc:.2f}%)")
        return acc

    def __segment(self, image_path: str) -> list:
        image: Image.Image = Image.open(image_path)
        segmented_digits, _ = ImageProcessor.segment(image, preprocess=True)
        for i in range(len(segmented_digits)):
            segmented_digits[i] = segmented_digits[i].convert("RGB")
            segmented_digits[i] = baseTransform(segmented_digits[i])
        return segmented_digits

    def __gen_output(self, image: torch.Tensor, _: int) -> torch.Tensor:
        return self.__model(image)
