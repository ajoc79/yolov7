import matplotlib.pyplot as plt
import torch
import cv2
import yaml
from torchvision import transforms
import numpy as np

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

street_img = cv2.imread('yolov7/inference/images/bus.jpg')
street_img = cv2.cvtColor(street_img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(12, 6))
plt.imshow(street_img)