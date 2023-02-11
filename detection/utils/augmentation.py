from typing import Tuple, Union, List, Dict

import cv2
import random
import hashlib 
import albumentations as A
import numpy as np

from . import bbox
from . import image
from PIL import Image as PImage
from pathlib import Path


class image_augment:
    default = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomCrop(width=550, height=550, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.8, contrast_limit=0.8, p=0.5),
        A.ChannelShuffle(p=0.5),
        ]

class augment:
    def __init__(
        self,
        img_aug:List=[],
        bbox_aug:A.BboxParams=None,
        bbox_format:str="yolo",
        resize:Union[List, Tuple]=(640, 640),
        ) -> None:
        # Declare an augmentation pipeline
        bbox_format_choice = ["coco", "pascal_voc", "albumentations", "yolo"]
        assert bbox_format in bbox_format_choice, f"bbox_format must be one of this {bbox_format_choice}."
        if len(img_aug) == 0:
            img_aug = image_augment.default
        if bbox_aug is None:
            bbox_aug = A.BboxParams(format=bbox_format, min_area=1024, min_visibility=0.1)
        if resize:
            img_aug.append(A.Resize(height=resize[1], width=resize[0], p=1))
        self.set_transform(img_aug=img_aug, bbox_aug=bbox_aug)            
        
    def set_transform(self, img_aug=None, bbox_aug=None):
        self.transform = A.Compose(transforms=img_aug, bbox_params=bbox_aug)
        return self.transform

    def process(
        self,
        img:np.array,
        bboxes:Union[List, np.array],
        class_labels:Union[List, np.array],
        ) -> Dict:
        self.transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
        return self.transformed

    def saved(self, saved_dir, filename_org):
        filename = f"{filename_org}_{hashlib.sha256(self.transformed['image']).hexdigest()}"
        img_path = Path(saved_dir).joinpath("images")
        img_path.mkdir(parents=False, exist_ok=True)
        img_anno = PImage.fromarray(self.transformed["image"])
        img_anno.save(img_path.joinpath(f"{filename}.jpg"))

        label_path = Path(saved_dir).joinpath("labels")
        label_path.mkdir(parents=False, exist_ok=True)
        bboxes = np.array(self.transformed["bboxes"])
        if self.transformed["bboxes"]:
            bboxes = np.hstack((bboxes[:,4].reshape(-1,1), bboxes[:,:4]))
        np.savetxt(label_path.joinpath(f"{filename}.txt"), bboxes, fmt="%10.8f", delimiter=" ")

        
def horizontal_shift(img:np.ndarray, bboxes:np.ndarray, value:int) -> Tuple[np.ndarray, np.ndarray]:
    img = image.horizontal_shift(img, value)
    bboxes = bbox.shift(bboxes, img, value, horizontal=True)
    return img, bboxes

def vertical_shift(img:np.ndarray, bboxes:np.ndarray, value:int):
    img = image.vertical_shift(img, value)
    bboxes = bbox.shift(bboxes, img, value, horizontal=False)
    return img, bboxes

def rotation(
    img:np.ndarray, 
    bboxes:np.ndarray, 
    angle:float, 
    random:bool=False, 
    crop:bool=True
    ) -> Tuple[np.ndarray, np.ndarray]:
    if random:
        angle = int(random.uniform(-angle, angle))
    img = image.rotation(img, angle, crop)
    bboxes = bbox.rotation(bboxes, img, angle, crop)
    return img, bboxes

def flip_horizontal(img:np.ndarray, bboxes:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img = image.flip(img, horizontal=True, vertical=False)
    bboxes = bbox.flip_horizontal(bboxes, img)
    return img, bboxes
    
def flip_vertical(img:np.ndarray, bboxes:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img = image.flip(img, horizontal=False, vertical=True)
    bboxes = bbox.flip_vertical(bboxes, img)
    return img, bboxes

def flip_both(img:np.ndarray, bboxes:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img = image.flip(img, horizontal=True, vertical=True)
    bboxes = bbox.flip_horizontal(bboxes, img)
    bboxes = bbox.flip_vertical(bboxes, img)
    return img, bboxes

def crop(img:np.ndarray, bboxes:np.ndarray, value:float, random:bool=False) -> Tuple[np.ndarray, np.ndarray]:
    if random:
        value = random.uniform(value, 1)
    img = image.crop(img, value)
    bboxes = bbox.crop(bboxes, img, value)
    return img, bboxes

def hue(img:np.ndarray, bboxes:np.ndarray, value:int, random:bool=False) -> Tuple[np.ndarray, np.ndarray]:
    if random:
        value = random.uniform(-value, value)
    return image.hue(img, value), bboxes

def brightness(img:np.ndarray, bboxes:np.ndarray, value:int, random:bool=False) -> Tuple[np.ndarray, np.ndarray]:
    if random:
        value = random.uniform(-value, value)
    return image.brightness(img, value), bboxes

def contrast(img:np.ndarray, bboxes:np.ndarray, value:float, random:bool=False) -> Tuple[np.ndarray, np.ndarray]:
    if random:
        value = random.uniform(-value, value)
    return image.contrast(img, value), bboxes

def noise(img:np.ndarray, bboxes:np.ndarray, noise_type:str) -> Tuple[np.ndarray, np.ndarray]:
    img = image.noise(img, noise_type)
    return img, bboxes