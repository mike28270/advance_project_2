from typing import List, Union

import cv2
import numpy as np

# def format_yolov5(targets: Union[List, np.ndarray], img: np.ndarray, label_pos: int=0) -> np.ndarray:
#     img_h, img_w, _ = img.shape
#     targets = np.array(targets)
#     ids = targets[:,label_pos].reshape(-1, 1)
#     bboxes = targets[:,1:5]
#     bboxes = np.apply_along_axis(convert_whRatio_to_xy, axis=1, arr=bboxes, img_w=img_w, img_h=img_h)
#     targets = np.hstack((bboxes, ids))
#     return targets

def format_yolov5(targets: Union[List, np.ndarray], img: np.ndarray, label_pos: int=0) -> np.ndarray:
    img_h, img_w, _ = img.shape
    targets = np.array(targets)
    if len(targets.shape) == 1:
        targets = targets.reshape(1,-1)
    targets[:,1:5] = np.apply_along_axis(convert_whc2xy, axis=1, arr=targets[:,1:5])
    targets[:,1:5] = np.apply_along_axis(convert_abs2rel, axis=1, arr=targets[:,1:5], img_w=img_w, img_h=img_h)
    ids = targets[:,label_pos].copy()
    bboxes = targets[:,1:5].copy()
    targets[:,0:4] = bboxes
    targets[:,4] = ids
    return targets

def convert_wh2xy(bbox: Union[List, np.ndarray]) -> np.ndarray:
    """
    Convert from pascal_voc:[x, y, w, h] to COCO:[x1, y1, x2, y2]
    Arg: 
    - bbox: [x, y, w, h]
        x: x coordinate of the bbox at top-left
        y: y coordinate of the bbox at top-left
        w: width of the bbox
        h: hight of the bbox
    Return:
    - bbox: [x1, y1, x2, y2]
        x1: x coordinate of the bbox at top-left (x_min)
        y1: y coordinate of the bbox at top-left (y_min)
        x2: x coordinate of the bbox at bottom-right (x_max)
        y2: y coordinate of the bbox at bottom-right (y_max)
    """
    assert len(bbox) == 4, "bbox must have the length equals to 4."
    x, y, w, h = bbox
    return [x, y, x+w, y+h]

def convert_wh2whc(bbox: Union[List, np.ndarray]) -> np.ndarray:
    """
    Convert from pascal_voc:[x, y, w, h] to YOLO:[xc, yc, w, h]
    Arg:
    - bbox: [x, y, w, h]
        x: x coordinate of the bbox at top-left
        y: y coordinate of the bbox at top-left
        w: width of the bbox
        h: hight of the bbox
    Return:
    - bbox: [xc, yc, w, h]
        xc: x coordinate of the bbox at center
        yc: y coordinate of the bbox at center
        w: width of the bbox
        h: hight of the bbox
    """
    assert len(bbox) == 4, "bbox must have the length equals to 4."
    x, y, w, h = bbox
    return [x+(w/2), y+(h/2), w, h]

def convert_xy2whc(bbox: Union[List, np.ndarray]) -> np.ndarray:
    """
    Convert from COCO:[x1, y1, x2, y2] to YOLO:[xc, yc, w, h]
    Arg:
    - bbox: [x1, y1, x2, y2]
        x1: x coordinate of the bbox at top-left (x_min)
        y1: y coordinate of the bbox at top-left (y_min)
        x2: x coordinate of the bbox at bottom-right (x_max)
        y2: y coordinate of the bbox at bottom-right (y_max)
    Return:
    - bbox: [xc, yc, w, h]
        xc: x coordinate of the bbox at center
        yc: y coordinate of the bbox at center
        w: width of the bbox
        h: hight of the bbox
    """
    assert len(bbox) == 4, "bbox must have the length equals to 4."
    x1, y1, x2, y2 = bbox
    w, h = x2-x1, y2-y1
    return [x1+(w/2), y1+(h/2), w, h]

def convert_xy2wh(bbox: Union[List, np.ndarray]) -> np.ndarray:
    """
    Convert from COCO:[x1, y1, x2, y2] to pascal_voc:[x, y, w, h]
    Arg:
    - bbox: [x1, y1, x2, y2]
        x1: x coordinate of the bbox at top-left (x_min)
        y1: y coordinate of the bbox at top-left (y_min)
        x2: x coordinate of the bbox at bottom-right (x_max)
        y2: y coordinate of the bbox at bottom-right (y_max)
    Return:
    - bbox: [x, y, w, h]
        x: x coordinate of the bbox at top-left
        y: y coordinate of the bbox at top-left
        w: width of the bbox
        h: hight of the bbox
    """
    assert len(bbox) == 4, "bbox must have the length equals to 4."
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2-x1, y2-y1]

def convert_whc2xy(bbox: Union[List, np.ndarray]) -> np.ndarray:
    """
    Convert from YOLO:[xc, yc, w, h] to COCO:[x1, y1, x2, y2]
    Arg:
    - bbox: [xc, yc, w, h]
        xc: x coordinate of the bbox at center
        yc: y coordinate of the bbox at center
        w: width of the bbox
        h: hight of the bbox
    Return:
    - bbox: [x1, y1, x2, y2]
        x1: x coordinate of the bbox at top-left (x_min)
        y1: y coordinate of the bbox at top-left (y_min)
        x2: x coordinate of the bbox at bottom-right (x_max)
        y2: y coordinate of the bbox at bottom-right (y_max)
    """
    assert len(bbox) == 4, "bbox must have the length equals to 4."
    xc, yc, w, h = bbox
    return [xc-(w/2), yc-(h/2), xc+(w/2), yc+(h/2)]

def convert_whc2wh(bbox: Union[List, np.ndarray]) -> np.ndarray:
    """
    Convert from YOLO:[xc, yc, w, h] to pascal_voc:[x, y, w, h]
    Arg:
    - bbox: [xc, yc, w, h]
        xc: x coordinate of the bbox at center
        yc: y coordinate of the bbox at center
        w: width of the bbox
        h: hight of the bbox
    Return:
    - bbox: [x, y, w, h]
        x: x coordinate of the bbox at top-left
        y: y coordinate of the bbox at top-left
        w: width of the bbox
        h: hight of the bbox
    """
    assert len(bbox) == 4, "bbox must have the length equals to 4."
    xc, yc, w, h = bbox
    return [xc-(w/2), yc-(h/2), w, h]

def convert_abs2rel(bbox: Union[List, np.ndarray], img_w:int, img_h:int) -> np.ndarray:
    """ Convert from absolute to relative position """
    return [bbox[0]*img_w, bbox[1]*img_h ,bbox[2]*img_w ,bbox[3]*img_h]

def convert_rel2abs(bbox: Union[List, np.ndarray], img_w:int, img_h:int) -> np.ndarray:
    """ Convert from relative to absolute position """
    return [bbox[0]/img_w, bbox[1]/img_h ,bbox[2]/img_w ,bbox[3]/img_h]

def convert_whRatio_to_xy(bbox: Union[List, np.ndarray], img_w:int, img_h:int) -> np.ndarray:
    x_r, y_r, w_r, h_r = bbox
    x1 = (x_r - w_r/2) * img_w
    y1 = (y_r - h_r/2) * img_h
    x2 = (x_r + w_r/2) * img_w
    y2 = (y_r + h_r/2) * img_h
    return [x1, y1, x2, y2]

def convert_xy_to_whRatio(bbox: Union[List, np.ndarray], img_w:int, img_h:int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    x_r = (x1 + x2) / (2 * img_w)
    y_r = (y1 + y2) / (2 * img_h)
    w_r = (x2 - x1) / img_w
    h_r = (y2 - y1) / img_h
    return [x_r, y_r, w_r, h_r]

def convert_wh_to_xy(bbox: Union[List, np.ndarray], img_w:int, img_h:int) -> np.ndarray:
    x, y, w, h = bbox
    x1 = x - (w / 2)
    y1 = y - (h / 2)
    x2 = x + (w / 2)
    y2 = y + (h / 2)
    return [x1, y1, x2, y2]

def convert_xy_to_wh(bbox: Union[List, np.ndarray], img_w:int, img_h:int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    x, y = x1, y1
    w = x2 - x1
    h = y2 - y2
    return [x, y, w, h]

def convert_whRatio_to_wh(bbox: Union[List, np.ndarray], img_w:int, img_h:int) -> np.ndarray:
    x_r, y_r, w_r, h_r = bbox
    x = x_r * img_w
    y = y_r * img_h
    w = w_r * img_w
    h = h_r * img_h
    return [x, y, w, h]

def convert_wh_to_whRatio(bbox: Union[List, np.ndarray], img_w:int, img_h:int) -> np.ndarray:
    x, y, w, h = bbox
    x_r = x / img_w
    y_r = y / img_h
    w_r = w / img_w
    h_r = h / img_h
    return [x_r, y_r, w_r, h_r]

def convert_xy_to_coor(bboxes: np.ndarray) -> np.ndarray:
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    x1, y1 = bboxes[:,0].reshape(-1,1), bboxes[:,1].reshape(-1,1)
    x2, y2 = x1 + width, y1
    x3, y3 = x1, y1 + height
    x4, y4 = bboxes[:,2].reshape(-1,1), bboxes[:,3].reshape(-1,1)    
    return np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))

def shift(bboxes:np.ndarray, img:np.ndarray, value:int, horizontal:bool =True) -> np.ndarray:
    def check_box(box):
        if box[a_indice[0]] == 0 and box[a_indice[1]] == 0:
            return np.array([np.nan]*len(box))
        else:
            return box

    h, w = img.shape[:2]

    if horizontal:
        a_indice = [0, 2]
        max_limit = w
    else:
        a_indice = [1, 3]
        max_limit = h
    
    _x = bboxes[:,a_indice] + value
    _x = np.where(_x > 0, _x, 0)
    _x = np.where(_x < max_limit, _x, max_limit)
    bboxes[:,a_indice] = _x
    bboxes = np.apply_along_axis(check_box, 1, bboxes)
    bboxes = bboxes[~np.isnan(bboxes).any(axis=1)]
    return bboxes

def rotation(bboxes:np.ndarray, img:np.ndarray, angle:float, crop:bool=True) -> np.ndarray:
    h, w = img.shape[:2] 
    cx, cy = (int(w / 2), int(h / 2))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    if not crop:
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        newW, newH = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
        M[0, 2] += (newW / 2) - cx
        M[1, 2] += (newH / 2) - cy
        w, w = newW, newH

    coor = convert_xy_to_coor(bboxes).reshape(-1,2)
    coor = np.hstack((coor, np.ones((coor.shape[0],1), dtype = type(coor[0][0]))))
    coor_rot = np.dot(M, coor.T).T.reshape(-1,8)

    # Limit min, max
    _x, _y = coor_rot[:,[0,2,4,6]], coor_rot[:,[1,3,5,7]]
    _x, _y = np.where(_x > 0, _x, 0), np.where(_y > 0, _y, 0)
    _x, _y = np.where(_x < w, _x, w), np.where(_y < h, _y, h)

    xmin = np.min(_x,1).reshape(-1,1)
    ymin = np.min(_y,1).reshape(-1,1)
    xmax = np.max(_x,1).reshape(-1,1)
    ymax = np.max(_y,1).reshape(-1,1)
    label = np.array(bboxes[:,4]).reshape(-1,1)
    
    return np.hstack((xmin, ymin, xmax, ymax, label))

def flip_horizontal(bboxes:np.ndarray, img:np.ndarray) -> np.ndarray:
    img_center = np.array(img.shape[:2])[::-1]/2
    img_center = np.hstack((img_center, img_center))
    bboxes[:,[0,2]] += 2*(img_center[[0,2]] - bboxes[:,[0,2]])
    box_w = abs(bboxes[:,0] - bboxes[:,2])
    bboxes[:,0] -= box_w
    bboxes[:,2] += box_w
    return bboxes

def flip_vertical(bboxes:np.ndarray, img:np.ndarray) -> np.ndarray:
    img_center = np.array(img.shape[:2])[::-1]/2
    img_center = np.hstack((img_center, img_center))
    bboxes[:,[1,3]] += 2*(img_center[[1,3]] - bboxes[:,[1,3]])
    box_w = abs(bboxes[:,1] - bboxes[:,3])
    bboxes[:,1] -= box_w
    bboxes[:,3] += box_w
    return bboxes

def crop(bboxes:np.ndarray, img:np.ndarray, value:float) -> np.ndarray:
    def crop_box(box):
        x1, y1, x2, y2, *label = box
        h, w = y2-y1, x2-x1
        cx, cy = x1+int(w/2), y1+int(h/2)
        newH, newW = int(h*(1+value)), int(w*(1+value))
        x1, y1 = cx - int(newW/2), cy + int(newH/2)
        x2, y2 = cx + int(newW/2), cy - int(newH/2)
        return x1, y1, x2, y2, *label
    assert 0. <= value <= 1., "Value should be between 0 and 1"
    h, w = img.shape[:2]
    newH, newW = int(h*value), int(w*value)
    bboxes[:,:4] = np.apply_along_axis(convert_xy_to_whRatio, axis=1, arr=bboxes[:,:4], img_w=w, img_h=h)
    bboxes[:,:4] = np.apply_along_axis(convert_whRatio_to_xy, axis=1, arr=bboxes[:,:4], img_w=w, img_h=h)
    print(bboxes)
    # bboxes = np.apply_along_axis(crop_box, 1, bboxes)
    # box_w = abs(bboxes[:,1] - bboxes[:,3])
    # bboxes[:,1] -= box_w
    # bboxes[:,3] += box_w
    return bboxes