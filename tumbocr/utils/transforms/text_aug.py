import random
import numpy as np
from .text_image_aug.augment import tia_distort, tia_stretch, tia_perspective

def text_aug(image,is_distort=True,is_stretch=True,is_perspective=True):
    new_img = image
    prob = 0.4
    if is_distort:
        img_height, img_width = image.shape[0:2]
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            try:
                new_img = tia_distort(new_img, random.randint(3, 6))
            except:
                print("Exception occured during tia_distort, pass it...")

    if is_stretch:
        img_height, img_width = image.shape[0:2]
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            try:
                new_img = tia_stretch(new_img, random.randint(3, 6))
            except:
                print("Exception occured during tia_stretch, pass it...")

    if is_perspective:
        if random.random() <= prob:
            try:
                new_img = tia_perspective(new_img)
            except:
                print("Exception occured during tia_perspective, pass it...")
    return new_img