from typing import Union, List, Tuple

import os
import yaml
import random
import shutil
import numpy as np
import pandas as pd
import albumentations as A

import time
from tqdm import tqdm
from PIL import Image as PImage
from pathlib import Path, PurePath
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from detection.utils.augmentation import augment


def readLabel_yolo(label_dir:str, class_names:List, label_file_ext:str="txt", skipemptyfile:bool=False) -> pd.DataFrame:
    labels = sorted(Path(label_dir).glob(f"*.{label_file_ext}"))
    assert len(labels) > 0, "No files in directory."
    df_list = []
    for label in labels:
        columns = ["cat", "x", "y", "w", "h"]
        if os.stat(label).st_size != 0:
            df_temp = pd.read_csv(label, header=None, names=columns, sep=" ")
            df_temp["filename"] = label.stem
            df_list.append(df_temp)
        elif skipemptyfile:
            df_temp = pd.DataFrame([[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]], columns=columns)
            df_temp["filename"] = label.stem
            df_list.append(df_temp)
        elif os.stat(label).st_size == 0:
            df_temp = pd.DataFrame([[len(class_names), 0.1, 0.1, 0.1, 0.1]], columns=columns)
            df_temp["filename"] = label.stem
            df_list.append(df_temp)
    # if len(df_list) > 0:
    df_label = pd.concat(df_list, axis=0, ignore_index=True)
    df_label = df_label[["filename", "x", "y", "w", "h", "cat"]]
    return df_label

def copyClass(df, class_id, copies=1):
    df_nothing = df[df["cat"]==class_id]
    df_list = [df_nothing for i in range(copies)]
    df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    return df

def split(
    df_label:pd.DataFrame,
    dataset_dir:str,
    val_size = 0.2,
    test_size = 0.1,
    stratify:bool=True,
    saved:bool=True,
    saved_dir:str=".",) -> dict:
    
    def createDir(root:str, dir:str):
        dir = Path(root).joinpath(dir)
        image_dir = Path(dir).joinpath("images")
        label_dir = Path(dir).joinpath("labels")
        dir.mkdir(parents=False, exist_ok=True)
        image_dir.mkdir(parents=False, exist_ok=True)
        label_dir.mkdir(parents=False, exist_ok=True)
        return image_dir, label_dir

    def moveFiles(
        dataset_dir, 
        saved_root,
        saved_dir,
        filenames:List):
        src_image_dir = Path(dataset_dir).joinpath("images")
        src_label_dir = Path(dataset_dir).joinpath("labels")
        des_image_dir, des_label_dir = createDir(root=saved_root, dir=saved_dir)
        for filename in filenames:
            # shutil.move(src_image_dir.joinpath(f"{filename}.jpg"), des_image_dir.joinpath(f"{filename}.jpg"))
            # shutil.move(src_label_dir.joinpath(f"{filename}.txt"), des_label_dir.joinpath(f"{filename}.txt"))
            shutil.copy(src_image_dir.joinpath(f"{filename}.jpg"), des_image_dir.joinpath(f"{filename}.jpg"))
            shutil.copy(src_label_dir.joinpath(f"{filename}.txt"), des_label_dir.joinpath(f"{filename}.txt"))

    assert (val_size+test_size) < 1, "The summation of validation and test size must less than 1."
    X = df_label.drop("cat", axis=1)
    y = df_label["cat"]
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, shuffle=True, random_state=42, stratify=y_train)
    else: 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, shuffle=True, random_state=42)
    
    if saved:
        dataset_dir = Path(dataset_dir)

        moveFiles(dataset_dir, saved_dir, "train", X_train["filename"].unique())
        moveFiles(dataset_dir, saved_dir, "valid", X_val["filename"].unique())
        moveFiles(dataset_dir, saved_dir, "test", X_test["filename"].unique())
    
    return {"X_train": X_train, 
            "X_val": X_val, 
            "X_test": X_test, 
            "y_train": y_train, 
            "y_val": y_val, 
            "y_test": y_test}

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
        resize:Union[List, Tuple]=(640, 640),
        ) -> None:
        self.data = yaml.load(open(Path(data_path)), Loader=yaml.SafeLoader)
        self.class_names = self.data["names"]

        self.image_dir = Path(image_dir)

        self.label_dir = label_dir
        self.df_label = readLabel_yolo(
            label_dir=label_dir, 
            class_names=self.class_names, 
            label_file_ext=label_file_ext,
            skipemptyfile=False)
        self.arr_label = self.df_label.to_numpy()

        self.saved_dir = Path(saved_dir)
        self.image_file_ext = image_file_ext
        self.label_file_ext = label_file_ext
        self.label_format = label_format
        self.resize = resize

    def copy_org(self) -> bool:
        image_paths = list(self.image_dir.glob(f"*.{self.image_file_ext}"))
        for image_path in tqdm(image_paths, total=len(image_paths)):
            filename = image_path.stem
            img = np.array(PImage.open(image_path))
            bboxes = self.arr_label[np.where(self.arr_label[:,0] == filename)][:,1:]
            aug = augment(img_aug=[], bbox_aug=None, resize=self.resize)
            aug.process(img, bboxes, self.class_names)
            aug.saved(saved_dir=self.saved_dir, filename_org=filename)
    
    def _findDictUnique(self, a_arr:np.array):
        unique, counts = np.unique(a_arr, return_counts=True)
        checking_cnt = dict(zip(unique, counts))
        return checking_cnt

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
        checking_cnt = self._findDictUnique(arr_checking[:,5])
        counts =  np.array(list(checking_cnt.values()))
        assert all(limit >= counts), (
            f"The limit must more than maximum number of every class ({max(counts)}) but now it is {limit}.")
        pbar = tqdm(total=sum(limit-counts))
        while True:
            seconds = time.time()
            class_id_select = 5
            if  (seconds / 60) == 15:
                class_id_select = 4
            elif  (seconds / 60) == 30:
                class_id_select = 3
            elif  (seconds / 60) == 45:
                class_id_select = 2
            elif  (seconds / 60) == 60:
                class_id_select = 1

            # Random Class depending on the top 5 minimum classes.
            sorted_d = OrderedDict(sorted(checking_cnt.items(), key=lambda item: item[1]))
            min_class_ids = [k for k, v in sorted_d.items()][:class_id_select]
            min_class_id = random.choice(min_class_ids)
            # Random an image
            index = np.where(self.arr_label[:,5] == min_class_id)
            unique, counts = np.unique(self.arr_label[index][:,0], return_counts=True)
            filename = random.choice(unique)

            # Load image and bboxes
            img = np.array(PImage.open(f"{self.image_dir}/{filename}.{self.image_file_ext}"))
            bboxes = self.arr_label[np.where(self.arr_label[:,0] == filename)][:,1:]
            if any(bboxes[:, 4] == len(self.class_names)):
                bboxes = []

            # Transform the image
            aug = augment(bbox_aug=None, resize=self.resize)
            transformed = aug.process(img, bboxes, self.class_names)
            # aug.saved(saved_dir="detection/dataset/machine_preprocessed/whole", filename_org=filename)

            # Add filename to the transformed result: [x, y, w, h, cat] -> [filename, x, y, w, h, cat]
            if transformed["bboxes"]:
                # Need to use pd.DataFrame to deal with np.dtype problem :(
                bboxes_new = pd.DataFrame([[filename]]*len(transformed["bboxes"]), columns=["filename"])
                bboxes_new[["x", "y", "w", "h", "cat"]] = pd.DataFrame(transformed["bboxes"])
            else:
                bboxes_new = pd.DataFrame([[filename, 0, 0, 0, 0, len(self.class_names)-1]])
            bboxes_new = bboxes_new.to_numpy()

            # Check classes whether reach limit
            checking_cnt = self._findDictUnique(arr_checking[:,5])
            bboxes_new_cnt = self._findDictUnique(bboxes_new[:,5])
            
            ## Save image and bboxes by checking whether the transformed exceed the limit
            if all(bboxes_new_cnt.get(k, 0) + checking_cnt.get(k, 0) <= limit for k in bboxes_new_cnt):
                aug.saved(saved_dir=self.saved_dir, filename_org=filename)
                arr_checking = np.vstack((arr_checking, bboxes_new))
                pbar.update(1)

            ## Break if all class exceed the limit
            if all(value > limit for value in checking_cnt.values()):
                pbar.close()
                break

