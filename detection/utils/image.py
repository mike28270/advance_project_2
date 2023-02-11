from typing import List, Union

import cv2
import numpy as np
import matplotlib.colors as mcolors
from detection.utils.bbox import convert_wh2xy, convert_whc2xy, convert_abs2rel

def drawBox(
    img:np.ndarray, 
    targets:Union[List, np.ndarray], 
    labels:Union[List, np.ndarray]=None,
    font:int = cv2.FONT_HERSHEY_SIMPLEX,
    colors:List = list(mcolors.CSS4_COLORS.values()),
    box_thickness:int = 3,
    text_thickness:int = 3,
    text_size:int = 3,
    targets_type:str = "abs:coco",
    ) -> np.ndarray: 
    h, w = img.shape[:2]
    targets = np.array(targets)

    if len(targets) != 0:
        if targets_type.split(":")[1] == "pascal_voc":
            targets[:,0:4] = targets[:,0:4]
        elif targets_type.split(":")[1] == "coco":
            targets[:,0:4] = [convert_wh2xy(bbox) for bbox in targets[:,0:4]]
        elif targets_type.split(":")[1] == "yolo":
            targets[:,0:4] = [convert_whc2xy(bbox) for bbox in targets[:,0:4]]
        else:
            raise "The target type must be 'coco', 'pascal' or 'yolo'"
        

        if targets_type.split(":")[0] == "rel":
            targets[:,0:4] = targets[:,0:4]
        elif targets_type.split(":")[0] == "abs":
            targets[:,0:4] = [convert_abs2rel(bbox, w, h) for bbox in targets[:,0:4]]
        else:
            raise "The target type must be either 'abs' or 'rel'"
    
        for i, (x1, y1, x2, y2, a_text) in enumerate(targets):
            color = mcolors.to_rgb(colors[i])*np.array((255, 255, 255))
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=box_thickness)
            if labels is None:
                a_string = f"{a_text:.0f}"
            else:
                a_string = f"{labels[int(a_text)]}"        
            cv2.putText(
                img, a_string, (int(x1+10), int(y1+text_size*30)), font, 
                text_size, color=color, thickness=text_thickness, lineType=cv2.LINE_AA)
    return img
        
def horizontal_shift(img:np.ndarray, value:int) -> np.ndarray:
    h, w = img.shape[:2]
    assert -w < value < w, f"Value should be between -{w} and greater than {w}"
    M = np.float32([[1,0,value],[0,1,0]])
    img = cv2.warpAffine(img, M, (w, h))
    return img

def vertical_shift(img:np.ndarray, value:int) -> np.ndarray:
    h, w = img.shape[:2]
    assert -h < value < h, f"Value should be between -{h} and greater than {h}"
    M = np.float32([[1,0,0],[0,1,value]])
    img = cv2.warpAffine(img, M, (w, h))
    return img

def rotation(img:np.ndarray, angle:float, crop:bool=True) -> np.ndarray:
    h, w = img.shape[:2] 
    cx, cy = (int(w / 2), int(h / 2))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    if crop:
        return cv2.warpAffine(img, M, (w, h))
    else:
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        newW = int((h * sin) + (w * cos))
        newH = int((h * cos) + (w * sin))
        M[0, 2] += (newW / 2) - cx
        M[1, 2] += (newH / 2) - cy
        return cv2.warpAffine(img, M, (newW, newH))

def flip(img:np.ndarray, horizontal:bool=True, vertical:bool=True) -> np.ndarray:
    if horizontal and vertical:
        return cv2.flip(img, -1)
    elif horizontal:
        return cv2.flip(img, 1)
    elif vertical:
        return cv2.flip(img, 0)
    else:
        return img

def crop(img:np.ndarray, value:float) -> np.ndarray:
    assert 0. <= value <= 1., "Value should be between 0 and 1"
    h, w = img.shape[:2]
    newH, newW = int(h*value), int(w*value)
    cx, cy = int(w/2), int(h/2)
    xstart, xend = abs(cx-int(newW/2)), abs(cx+int(newW/2))
    ystart, yend = abs(cy-int(newH/2)), abs(cy+int(newH/2))
    img = cv2.resize(img[ystart:yend, xstart:xend, :], (w, h))
    return img

def hue(img:np.ndarray, value:int) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def brightness(img:np.ndarray, value:int) -> np.ndarray:
    assert 0 <= value <= 100, "Value should be between 0 and greater than 100"
    return cv2.convertScaleAbs(img, beta=value)

def contrast(img:np.ndarray, value:float) -> np.ndarray:
    assert 1.0 <= value <= 3.0, "Value should be between 1.0 and greater than 3.0"
    return cv2.convertScaleAbs(img, alpha=value)

def noise(img:np.ndarray, noise_type:str) -> np.ndarray:
    # speckle is only option that's currently working
    check_list = ["gauss", "s&p", "poisson", "speckle"]
    assert noise_type in check_list, f"The type should be one in check list: {check_list}"
    if noise_type == "gauss":
        row,col,ch= img.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = img + gauss
        print(noisy)
        return noisy
    elif noise_type == "s&p":
        row,col,ch = img.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in img.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in img.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        row,col,ch = img.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = img + img * gauss
        return noisy
