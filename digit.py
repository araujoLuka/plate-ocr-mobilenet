"""Module to extract an alphanumerical digit from a license plate image"""

__all__ = ["Digit"]
__author__ = "Lucas C. Araujo"
__version__ = "1.0.0"

# Import the necessary libraries
from PIL import Image
import argparse
import os

class Digit:
    def __init__(self, image_path: str, index: int, plate_length: int = 7, has_hyphen: bool = True):
        if index < 1 or index > 7:
            raise ValueError("Index must be between 1 and max length"
                                + f"({plate_length})"
                                + "\nType --help for more information")
        
        if plate_length < 0:
            raise ValueError("Plate length must be greater than 0")
          
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"{image_path} does not exist")
          
        if not image_path.endswith(".jpg") and not image_path.endswith(".jpeg") and not image_path.endswith(".png"):
            raise ValueError("Image must be in jpg, jpeg or png format")
          
        self.__image_path: str = image_path
        self.__index: int = index
        self.__plate_length: int = plate_length
        self.__has_hyphen: bool = has_hyphen
        self.__digit: Image.Image = self.__extract_digit()
    
    @property
    def digit(self) -> Image.Image:
        return self.__digit

    def save(self, 
             file_path: str = "",
             directory: str = "./",
             override: bool = False,
             format: str = "jpeg",
             gen_path: bool = True) -> None:
        """Save the digit to a file"""
        if file_path == "" and not gen_path:
            raise ValueError("Please provide a file path")
        
        self.__check_directory(directory)
        
        file_path = os.path.join(directory, file_path) + f".{format}"
        if gen_path:
            file_path = self.__generated_path(directory=directory, format=format)
        
        if not override and os.path.exists(file_path):
            raise FileExistsError(f"{file_path} already exists")
        
        self.__digit.save(file_path, format=format)

    def show(self) -> None:
        """Show the digit on the screen"""
        self.__digit.show()
       
    def __extract_digit(self) -> Image.Image:
        """Access the image, transform it and return the digit
        
        The digit is cropped from the license plate based on:
            - index: the index of the digit (1-7)
            - plate_length: the plate_length number of digits in the license plate
            - has_hyphen: whether the license plate has a hyphen or not
        """
        image = Image.open(self.__image_path)
        width, height = image.size

        # Rotate if the image is in portrait mode
        if height > width:
            image = image.rotate(90, expand=True)
            width, height = image.size

        # Calculate the width of the digit
        digit_width = width // (self.__plate_length + 1)
        if not self.__has_hyphen:
            digit_width = width // self.__plate_length
        
        # Adjust the index if the license plate has a hyphen
        if self.__index >= 4 and self.__has_hyphen:
            self.__index += 1
        
        image = image.crop(((self.__index - 1) * digit_width, 0, self.__index * digit_width, height))

        return image

    def __check_directory(self, directory: str = "./") -> None:
        """Check if the directory exists, if not create it"""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def __gen_filename(self, directory: str = "./", plate_name: str = "", format: str = "jpg", i: int = -1) -> str:
        """Generate a file name for the digit"""
        if plate_name == "":
            plate_name = self.__image_path.split("/")[-1].split(".")[0]
        plate_digits = [d for d in plate_name if d != "-"]
        file_name = f"{plate_digits[self.__index - 1]}_from_plate_{plate_name}"
        if i != -1: 
            file_name += f"_{i}"
        file_name += f".{format}"
        file_path = os.path.join(directory, file_name)

        return file_path

    def __generated_path(self, directory: str = "./", format: str = "jpg") -> str:
        """Return a generated path for the digit"""
        i = 2   # To create a unique file name, if needed
        plate_name = self.__image_path.split("/")[-1].split(".")[0]
        file_path = self.__gen_filename(directory, plate_name, format)

        while os.path.exists(file_path):
            file_path = self.__gen_filename(directory, plate_name, format, i)
            i += 1
        
        return file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get a digit from the license plate")
    parser.add_argument("image_path", type=str, help="Path to the image")
    parser.add_argument("index", type=int, help="Index of the digit (1-7)", nargs="?", default=1)
    parser.add_argument("--path", type=str, help="Path to save the digit", default="")
    parser.add_argument("--dir", type=str, help="Directory to save the digit", default="./")
    parser.add_argument("--show", action="store_true", help="Show the digit", default=False)
    parser.add_argument("--length", type=int, help="Total number of digits in the license plate", default=7)
    parser.add_argument("--no-hyphen", action="store_true", help="License plate does not have hyphen")
    parser.add_argument("--all", action="store_true", help="Get all digits", default=False)
    args = parser.parse_args()

    image_path = args.image_path
    index = args.index
    length = args.length
    has_hyphen = not args.no_hyphen

    max_index = index
    if args.all:
        index = 1
        max_index = length
    
    for i in range(index, max_index + 1):
        digit = Digit(image_path, i, plate_length=length, has_hyphen=has_hyphen)

        if args.show:
            digit.show()
        
        if args.all:
            digit.save(directory=args.dir)
        else:
            if args.path == "":
                digit.save(directory=args.dir)
            else:
                digit.save(file_path=args.path, directory=args.dir, gen_path=False, override=True)
