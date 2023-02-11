from typing import Union, List, Tuple

import os
import yaml
import random
import numpy as np
import pandas as pd
import albumentations as A

from tqdm import tqdm
from pathlib import Path
from PIL import Image as PImage
from sklearn.model_selection import train_test_split
from detection.utils.augmentation import augment


def readLabel_yolo(label_dir:str, class_names:List, label_file_ext:str="txt") -> pd.DataFrame:
    labels = sorted(Path(label_dir).glob(f"*.{label_file_ext}"))
    df_list =[]
    for label in labels:
        columns = ["cat", "x", "y", "w", "h"]
        if os.stat(label).st_size != 0:
            df_temp = pd.read_csv(label, header=None, names=columns, sep=" ")
        else:
            df_temp = pd.DataFrame([[len(class_names), 0.1, 0.1, 0.1, 0.1]], columns=columns)
        df_temp["filename"] = label.stem
        df_list.append(df_temp)
    df_label = pd.concat(df_list, axis=0, ignore_index=True)
    return df_label[["filename", "x", "y", "w", "h", "cat"]]

class Dataset:
    def __init__(
        self,
        data_path:str,
        image_dir:str,
        label_dir:str,
        saved_dir:str = ".",
        image_file_ext:str = "JPG",
        label_file_ext:str = "txt",
        label_format:str = "abs:yolo",
        ) -> None:
        self.data = yaml.load(open(Path(data_path)), Loader=yaml.SafeLoader)
        self.class_names = self.data["names"]

        self.image_dir = Path(image_dir)

        self.label_dir = label_dir
        self.df_label = readLabel_yolo(
            label_dir=label_dir, 
            class_names=self.class_names, 
            label_file_ext=label_file_ext)
        self.arr_label = self.df_label.to_numpy()

        self.saved_dir = Path(saved_dir)
        self.image_file_ext = image_file_ext
        self.label_file_ext = label_file_ext
        self.label_format = label_format

    def copy_org(self, resize:Union[None, List]=None) -> bool:
        image_paths = list(self.image_dir.glob(f"*.{self.image_file_ext}"))
        for image_path in tqdm(image_paths, total=len(image_paths)):
            filename = image_path.stem
            img = np.array(PImage.open(image_path))
            bboxes = self.arr_label[np.where(self.arr_label[:,0] == filename)][:,1:]
            aug = augment(resize=resize)
            aug.process(img, bboxes, self.class_names)
            aug.saved(saved_dir=self.saved_dir, filename_org=filename)

    def balance_class(self, limit:int, target_dir:Path=None) -> bool:
        if target_dir is None:
            target_dir = self.saved_dir
        else:
            target_dir = Path(target_dir)
        df_checking = readLabel_yolo(
            label_dir=target_dir.joinpath("labels"), 
            class_names=self.class_names, 
            label_file_ext=self.label_file_ext)
        arr_checking = df_checking.to_numpy()
        unique, counts = np.unique(arr_checking[:,5], return_counts=True)
        assert all(limit > counts), (
            f"The limit must more than maximum number of every class ({max(counts)}) but now it is {limit}.")
        pbar = tqdm(total=sum(limit-counts))
        while True:
            # Random an image
            filename = random.choice(self.arr_label[:,0])

            # Load image and bboxes
            img = np.array(PImage.open(f"{self.image_dir}/{filename}.jpg"))
            bboxes = self.arr_label[np.where(self.arr_label[:,0] == filename)][:,1:]

            # Transform the image
            aug = augment()
            transformed = aug.process(img, bboxes, self.class_names)
            # aug.saved(saved_dir="detection/dataset/machine_preprocessed/whole", filename_org=filename)

            # Add filename to the transformed result: [x, y, w, h, cat] -> [filename, x, y, w, h, cat]
            dtype = "|S10, f, f, f, f, f"
            if transformed["bboxes"]:
                # Use pd.DataFrame to deal with np.dtype problem
                bboxes_new = pd.DataFrame([[filename]]*len(transformed["bboxes"]), columns=["filename"])
                bboxes_new[["x", "y", "w", "h", "cat"]] = pd.DataFrame(transformed["bboxes"])
            else:
                bboxes_new = pd.DataFrame([[filename, 0, 0, 0, 0, len(self.class_names)-1]])
            bboxes_new = bboxes_new.to_numpy()

            # Check classes whether reach limit
            unique, counts = np.unique(arr_checking[:,5], return_counts=True)
            checking_cnt = dict(zip(unique, counts))
            unique, counts = np.unique(bboxes_new[:,5], return_counts=True)
            bboxes_new_cnt = dict(zip(unique, counts))
            
            ## Save image and bboxes by checking whether the transformed exceed the limit
            if all(bboxes_new_cnt.get(k, 0) + checking_cnt.get(k, 0) <= limit for k in bboxes_new_cnt):
                aug.saved(saved_dir=self.saved_dir, filename_org=filename)
                arr_checking = np.vstack((arr_checking, bboxes_new))
                pbar.update(1)

            ## Break if all class exceed the limit
            if all(value > limit for value in checking_cnt.values()):
                pbar.close()
                break