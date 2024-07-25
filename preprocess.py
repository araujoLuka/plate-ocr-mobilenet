import os
from PIL import Image

from utils.image import ImageProcessor
from utils.progress import progress_bar

if __name__ == "__main__":
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)

    print("Preprocessing validation data...")
    val_data_dir: str = script_dir + "/assets/images/val/"
    output_dir: str = script_dir + "/assets/images/val_preprocessed/"

    os.makedirs(output_dir, exist_ok=True)

    for i, filename in enumerate(os.listdir(val_data_dir)):
        progress_bar(i+1, len(os.listdir(val_data_dir)), print_it=True)
        img = Image.open(val_data_dir + filename)
        img = ImageProcessor.preprocess(img)
        img.save(output_dir + filename)
        img.close()

    print("Finished! Total images preprocessed: " + str(len(os.listdir(output_dir))))
