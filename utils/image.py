from dataclasses import dataclass
from PIL import Image, ImageFilter
from skimage.segmentation import flood_fill
import numpy as np
import cv2

@dataclass
class ImageDimension:
    xMin: np.intp
    xMax: np.intp
    yMin: np.intp
    yMax: np.intp

    def __str__(self):
        return f"xMin: {self.xMin}, xMax: {self.xMax},\nyMin: {self.yMin}, yMax: {self.yMax}"

class ImageProcessor:
    @staticmethod
    def binarize(image: Image.Image, threshold: int = 100) -> Image.Image:
        grayImage: Image.Image = image.convert("L")
        image = grayImage.point(lambda p: 0 if p < threshold else 255)
        return image

    @staticmethod
    def invert(image: Image.Image) -> Image.Image:
        # Check if image need to be inverted
        binaryImage = ImageProcessor.binarize(image, threshold=128)
        # Get the mean of the top lef square 3x3 pixels
        topLeftSquare = [binaryImage.getpixel((x, y)) for x in range(3) for y in range(3)]
        colorMean = np.mean(np.array(topLeftSquare))
        # Check if the mean is more than 128
        # If it is, the image is already inverted (black text on white background)
        if colorMean > 128:
            return image
        return image.point(lambda p: 255 - p)

    @staticmethod
    # Based on: https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv
    def normalize(image: Image.Image) -> Image.Image:
        binaryImage = ImageProcessor.binarize(image, threshold=128)
        topLeft = binaryImage.getpixel((0, 0))
        bottomRight = binaryImage.getpixel((binaryImage.size[0] - 1, binaryImage.size[1] - 1))
        if topLeft == bottomRight:
            return image

        rgb_planes = cv2.split(np.array(image))
        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = diff_img.copy()
            cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
            cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)
        
        cv2_norm = cv2.merge(result_norm_planes)
        result_norm = Image.fromarray(cv2_norm)
        return result_norm

    @staticmethod
    def gaussian_blur(image: Image.Image, radius: float = 0.5) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    @staticmethod
    def noise_reduction(image: Image.Image) -> Image.Image:
        image = ImageProcessor.gaussian_blur(image, radius=1)
        # image = image.filter(ImageFilter.MedianFilter(size=5))
        # image = image.filter(ImageFilter.Kernel((3, 3), [1, 1, 1, 1, 1, 1, 1, 1, 1], scale=9))
        image = image.filter(ImageFilter.RankFilter(size=3, rank=5))
        return image

    @staticmethod
    def sharpen(image: Image.Image) -> Image.Image:
        image = image.filter(ImageFilter.SMOOTH)
        image = image.filter(ImageFilter.EDGE_ENHANCE)
        image = image.filter(ImageFilter.UnsharpMask(radius=10, percent=150, threshold=50))
        # image = image.filter(ImageFilter.SHARPEN)
        return image

    @staticmethod
    def resize(image: Image.Image, width: int = 520, height: int = 520) -> Image.Image:
        # Maintain the aspect ratio
        if image.size[0] > image.size[1]:
            height = int((width / image.size[0]) * image.size[1])
        else:
            width = int((height / image.size[1]) * image.size[0])
        return image.resize((width, height), Image.Resampling.BICUBIC)

    @staticmethod
    def pad(image: Image.Image, size: int = 5) -> Image.Image:
        pad = Image.new("L", (image.size[0] + size, image.size[1] + size*2), 255)
        pad.paste(image, (size//2, size))
        return pad

    @staticmethod
    def preprocess(image: Image.Image) -> Image.Image:
        image = ImageProcessor.normalize(image)
        image = ImageProcessor.invert(image)
        image = ImageProcessor.sharpen(image)
        image = ImageProcessor.resize(image)
        image = ImageProcessor.noise_reduction(image)
        image = ImageProcessor.gaussian_blur(image)
        image = ImageProcessor.binarize(image)
        # image = ImageProcessor.pad(image)
        return image

    @staticmethod
    def __crop_segment(segment: np.ndarray, seg_dim: ImageDimension) -> np.ndarray:
        segment = segment[seg_dim.yMin:seg_dim.yMax, seg_dim.xMin:seg_dim.xMax]
        return segment

    @staticmethod
    def __resize_segment(segment: np.ndarray) -> np.ndarray:
        if segment.shape[0] < 40:
            diff = 40 - segment.shape[0]
            top = diff // 2
            bottom = diff - top
            segment = np.pad(segment, ((top, bottom), (0, 0)), mode="constant", constant_values=255)
        if segment.shape[1] < 30:
            diff = 30 - segment.shape[1]
            left = diff // 2
            right = diff - left
            segment = np.pad(segment, ((0, 0), (left, right)), mode="constant", constant_values=255)
        return segment

    @staticmethod
    def __extend_image(image: Image.Image, scalar: int) -> Image.Image:
        new_size = (image.size[0] * scalar, image.size[1] * scalar)
        return image.resize(new_size, Image.Resampling.BICUBIC)

    @staticmethod
    def show_image(image: Image.Image, pause: bool = False) -> None:
        image.show()
        if pause:
            input("Press Enter to continue...")

    @staticmethod
    def show_image_from_array(np_image: np.ndarray, pause: bool = False) -> None:
        image = Image.fromarray(np_image)
        ImageProcessor.show_image(image, pause)

    @staticmethod
    def __is_small_segment(segment: np.ndarray, marker: int) -> bool:
        return np.sum(segment == marker) < 30

    @staticmethod
    def __valid_segment(seg_dim: ImageDimension) -> bool:
        # Skip hyphen
        if seg_dim.yMax - seg_dim.yMin < 40 and seg_dim.xMax - seg_dim.xMin < 40:
            # print("Skip hyphen")
            # print(seg_dim)
            # input("Press Enter to continue...")
            return False

        # Skip big horizontal segments
        if seg_dim.xMax - seg_dim.xMin > 100:
            # print("Skip big horizontal segment")
            return False

        return True

    @staticmethod
    def __remove_segment_from_image(segment: np.ndarray, np_image: np.ndarray, marker: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove the segment from the image to avoid duplicate segments

        Also update the segment with the original image values

        Args:
            segment (np.ndarray): The segment to be removed
            np_image (np.ndarray): The image as numpy array
            marker (int): The marker of the segment

        Returns:
            np.ndarray: The updated image without the segment
            np.ndarray: The updated segment with the original image values
        """
        # Remove segment from image to avoid duplicate segments
        for x in range(segment.shape[1]):
            for y in range(segment.shape[0]):
                point = np_image[y, x]
                np_image[y, x] = 255 if segment[y, x] == marker else point
                segment[y, x] = point if segment[y, x] == marker else 255

        return np_image, segment

    @staticmethod
    def __get_dimension(segment: np.ndarray, offset: int = 0, marker: int = 128) -> ImageDimension:
        xMin = np.min(np.where(segment == marker)[1]) - offset
        xMax = np.max(np.where(segment == marker)[1]) + offset
        yMin = np.min(np.where(segment == marker)[0]) - offset
        yMax = np.max(np.where(segment == marker)[0]) + offset
        return ImageDimension(xMin, xMax, yMin, yMax)

    @staticmethod
    def __combine_with_near_segment(segment: np.ndarray, np_image: np.ndarray, marker: int, distance: int = 15) -> tuple[np.ndarray, bool]:
        """
        Combine the current segment with a near segment (on the right side of the current segment)
        
        First, search for the middle y of the segment and start searching to the right
        until a big segment is found. If a big segment is found, stop the search and
        add the founded segment to the current segment.
        
        Args:
            segment (np.ndarray): The current segment
            np_image (np.ndarray): The image as numpy array
            marker (int): The marker of the current segment
            distance (int, optional): The distance to search to the right. Defaults to 15.
        
        Returns:
            np.ndarray: The updated segment or the same segment if no near was found
            bool: True if a near segment was found, False otherwise
        """
        seg_dim = ImageProcessor.__get_dimension(segment, marker=marker)
        xStart = seg_dim.xMax
        xMaxSearch = xStart + distance
        xMid = (seg_dim.xMax + seg_dim.xMin) // 2
        y = (seg_dim.yMax + seg_dim.yMin) // 2
        near_marker = marker + 50
        # print(f"xStart: {xStart}, xMaxSearch: {xMaxSearch}, xMid: {xMid}")

        # print(f"Searching near segment close to: (x, y): ({xMid}, {y}")
        for x in range(xStart, xMaxSearch):
            if x >= np_image.shape[1]:
                break

            seg_copy = segment.copy()
            seg_copy[y, xMid:xStart] = 80
            seg_copy[y, xStart:xMaxSearch] = 200
            seg_copy[y, x] = 40

            if segment[y, x] == marker or np_image[y, x] == 255:
                continue

            # print("Near point found:", f"(x, y): ({x}, {y})")
            # print("Current value:", np_image[y, x])
            # print("Current segment value:", segment[y, x])
            # print("Current segcopy value:", seg_copy[y, x])

            near_segment = flood_fill(np_image, (y, x), near_marker)
            seg_copy = np.where(near_segment == near_marker, near_marker, seg_copy) # update seg_copy with near_segment
            # ImageProcessor.show_image_from_array(seg_copy, pause=True)

            if np.sum(near_segment == near_marker) > 30:
                segment = np.where(near_segment == near_marker, marker, segment)
                return segment, True

        return segment, False
    
    @staticmethod
    def segment(image: Image.Image, preprocess: bool = False) -> tuple[list[Image.Image],Image.Image]:
        digits: list = []

        if preprocess:
            image = ImageProcessor.preprocess(image)

        # Convert image to numpy array
        np_image = np.array(image) 

        # Flood fill to segment the image
        counter = 0
        for i in range(np_image.shape[1]):
            for j in range(np_image.shape[0]):
                if np_image[j, i] == 0:
                    seg_marker = 128 + counter
                    segment = flood_fill(np_image, (j, i), seg_marker)

                    if ImageProcessor.__is_small_segment(segment, seg_marker):
                        continue

                    seg_dim = ImageProcessor.__get_dimension(segment, offset=5, marker=seg_marker)
                    # print(f"Segment {counter} dimension:", seg_dim)

                    if not ImageProcessor.__valid_segment(seg_dim):
                        np_image, _ = ImageProcessor.__remove_segment_from_image(segment, np_image, marker=seg_marker)
                        continue

                    segment, found = ImageProcessor.__combine_with_near_segment(segment, np_image, seg_marker)
                    if found:
                        seg_dim = ImageProcessor.__get_dimension(segment, offset=5, marker=seg_marker) # Update segment dimension

                    np_image, segment = ImageProcessor.__remove_segment_from_image(segment, np_image, marker=seg_marker)
                    segment = ImageProcessor.__crop_segment(segment, seg_dim)
                    segment = ImageProcessor.__resize_segment(segment)
                    seg_image = Image.fromarray(segment)
                    seg_image = ImageProcessor.__extend_image(seg_image, scalar=2)

                    digits.append(seg_image)
                    counter += 1
        
        # print(f"Number of segments: {counter}")
        # print(f"Number of digits: {len(digits)}")
        
        # print(f"Shape of image: {np_image.shape}")
        # print(f"Shape of segment: {digits[0].size}")

        return digits, image
    
if __name__ == "__main__":
    images = []
    # images.append(Image.open("/home/lucas/Ufpr/2024_1/OCI/plate-ocr/assets/images/val/DZO-1054.jpg")) # Normal image
    # images.append(Image.open("/home/lucas/Ufpr/2024_1/OCI/plate-ocr/assets/images/val/IWM-3796.jpg")) # Noisy image
    # images.append(Image.open("/home/lucas/Ufpr/2024_1/OCI/plate-ocr/assets/images/val/BCX-9516.jpg")) # Blurry image
    # images.append(Image.open("/home/lucas/Ufpr/2024_1/OCI/plate-ocr/assets/images/val/MJU-4343.jpg")) # Bad image
    # images.append(Image.open("/home/lucas/Ufpr/2024_1/OCI/plate-ocr/assets/images/val/MIX-0878.jpg")) # Bad image
    # images.append(Image.open("/home/lucas/Ufpr/2024_1/OCI/plate-ocr/assets/images/val/UOF-2949.jpg")) # Bad image
    images.append(Image.open("/home/lucas/Ufpr/2024_1/OCI/plate-ocr/assets/images/val/SOK-1305.jpg")) # Good image
    # images.append(Image.open("/home/lucas/Downloads/custom.jpg")) # Custom image

    for image in images:
        ImageProcessor.show_image(image)
        segmented, preproc = ImageProcessor.segment(image, preprocess=True)
        ImageProcessor.show_image(preproc)
        for seg in segmented:
            ImageProcessor.show_image(seg, pause=True)
        if images.index(image) != len(images) - 1:
            input("Press Enter to continue...")
