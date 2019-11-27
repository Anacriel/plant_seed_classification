import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math
from glob import glob
from mpl_toolkits.axes_grid1 import ImageGrid
from collections import defaultdict
from skimage import morphology

plt.rcParams['font.size'] = 8


def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()


def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix


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


def plot_raw_grid(images, cols=12):
    fig = plt.figure(1, (8, 8))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(cols, cols),
                     axes_pad=0.05,  # pad between axes in inch.
                     )

    plt.rcParams['font.size'] = 8
    k = 0
    for label, val in images.items():
        n = 0
        for i in range(cols):
            ax = grid[k]
            if i == 0:
                ax.text(-1000, 112, label, verticalalignment='center')
            img = images[label][n]
            resized_img = cv2.resize(img, (240, 240))
            ax.imshow(resized_img)
            ax.axis('off')
            n = n + 1
            k = k + 1
    plt.show()


def plot_in_line(images, cols=3):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    for col_i in range(cols):
        img = images[col_i][:, :, ::-1]
        axs[col_i].imshow(img)
        axs[col_i].axis('off')


def create_mask_for_plant(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([24, 60, 0])  # second 75 or 60
    upper_green = np.array([150, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    imglab = morphology.label(mask)  # create labels in segmented image
    cleaned = morphology.remove_small_objects(imglab, min_size=64, connectivity=1)
    img3 = np.zeros(cleaned.shape)  # create array of size cleaned
    img3[cleaned > 0] = 255
    img3 = np.uint8(img3)

    return img3


def segment_labeled_plants(images, sharpen=False):
    seg_images = {}
    for label, val in images.items():
        seg_images[label] = []
        for n in range(len(images[label])):
            seg_image, mask = segment_plant(images[label][n])
            if sharpen:
                seg_image = sharpen_image(seg_image)

            seg_images[label].append(seg_image)

    return seg_images


def segment_plant(image):
    mask = create_mask_for_plant(image)
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    masked_img[mask == 0] = 255

    return masked_img, mask


def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

