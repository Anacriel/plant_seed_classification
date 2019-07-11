import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from glob import glob
from mpl_toolkits.axes_grid1 import ImageGrid
from collections import defaultdict
import skimage.segmentation as seg
import skimage.color as color

plt.rcParams['font.size'] = 8


def read_images(path, data_type='train'):
    total = 0
    if data_type == 'train':
        images = defaultdict(list)
        for class_dir_name in os.listdir(path):  # total images per class
            if '.DS_Store' == class_dir_name:
                continue

            class_dir_path = os.path.join(path, class_dir_name)
            class_label = class_dir_name

            for image_path in glob(os.path.join(class_dir_path, "*.png")):
                image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image_bgr = cv2.resize(image_bgr, (200, 200), interpolation=cv2.INTER_AREA)
                images[class_label].append(image_bgr)
                total = total + 1

        print(total)
        return images
    elif data_type == 'test':
        images = []
        titles = []

        for image_path in glob(os.path.join(path, "*.png")):
            titles.append(os.path.basename(image_path))
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_bgr = cv2.resize(image_bgr, (200, 200), interpolation=cv2.INTER_AREA)
            images.append(image_bgr)
            total = total + 1

        print(total)
        return images, titles


def plot_raw_grid(images, labels, cols=12):
    fig = plt.figure(1, (8, 8))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(cols, cols),
                     axes_pad=0.05)
    n = 0
    for str_i in range(cols):
        for col_i in range(cols):
            ax = grid[n]
            if col_i == 0:
                ax.text(-1000, 112, labels[n], verticalalignment='center')
            img = images[n]
            ax.imshow(img)
            ax.axis('off')
            str_i = str_i + 1
            n = n + 1


def create_mask_for_plant(image):
    blur = cv2.GaussianBlur(image, (3, 3), 2)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_green = np.array([22, 60, 0])  # second 75
    upper_green = np.array([150, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return closed_mask


def segment_plant(image):
    mask = create_mask_for_plant(image)
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    masked_img[mask == 0] = 255

    return masked_img


def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

