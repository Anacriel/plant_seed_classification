import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from glob import glob
from mpl_toolkits.axes_grid1 import ImageGrid
from collections import defaultdict
from skimage import morphology


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
            img = segment_plant(img)
            img = img[:, :, ::-1]
            ax.imshow(img)
            ax.axis('off')
            str_i = str_i + 1
            n = n + 1


def plot_in_line(images, labels, cols=3):
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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    imglab = morphology.label(closed_mask)  # create labels in segmented image
    cleaned = morphology.remove_small_objects(imglab, min_size=64, connectivity=1)
    img3 = np.zeros(cleaned.shape)  # create array of size cleaned
    img3[cleaned > 0] = 255
    img3 = np.uint8(img3)

    return img3


def segment_plant(image):
    mask = create_mask_for_plant(image)
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    masked_img[mask == 0] = 0

    return masked_img


def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

