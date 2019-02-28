import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from glob import glob
from mpl_toolkits.axes_grid1 import ImageGrid


def read_images(path):
    total = 0
    images = {}
    for class_dir_name in os.listdir(path):  # total images per class
        if '.DS_Store' != class_dir_name:
            class_dir_path = os.path.join(path, class_dir_name)
            class_label = class_dir_name

            images[class_label] = []
            for image_path in glob(os.path.join(class_dir_path, "*.png")):
                image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                images[class_label].append(image_bgr)
                total = total + 1

    print(total)
    return images


def plot_class(label, images, rows=3, cols=3):
    fig, axs = plt.subplots(rows, cols, figsize=(3, 3))
    n = 0
    for i in range(0, rows):
        for j in range(0, cols):
            axs[i, j].xaxis.set_ticklabels([])
            axs[i, j].yaxis.set_ticklabels([])
            axs[i, j].imshow(images[label][n])
            n += 1

    plt.suptitle(label)
    plt.show()


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
            #if i == cols - 1:
            #    ax.text(250, 112, label, verticalalignment='left')
            k = k + 1
    plt.show()


def resize_images(path, height=300, width=300):
    images = {}
    for class_dir_name in os.listdir(path):  # total images per class
        if '.DS_Store' != class_dir_name:
            class_dir_path = os.path.join(path, class_dir_name)
            for image_path in glob(os.path.join(class_dir_path, "*.png")):
                image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                resized_img = cv2.resize(image_bgr, (height, width))
                cv2.imwrite(image_path, resized_img)


data_dir = '../../data/raw/'
proc_data_dir = '../../data/processed/'

test_image_dir = os.path.join(data_dir, 'test')
train_image_dir = os.path.join(data_dir, 'train')


# Let's process resized pics (300x300px)
images_per_class = read_images(proc_data_dir + 'train')

# Called only once, don't need any more...
# resize_images(proc_data_dir + 'train')

# for key, value in images_per_class.items():
#     print("{} - {}".format(key, len(value)) + ' images')


plot_raw_grid(images_per_class)


sensitivity = 35
lower_hsv = np.array([60 - sensitivity, 100, 50])
upper_hsv = np.array([60 + sensitivity, 255, 255])


def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def segment_plant(images, sharpen=True):
    seg_images = {}
    for label, val in images.items():
        seg_images[label] = []
        for n in range(len(images[label])):
            mask = create_mask_for_plant(images[label][n])
            seg_image = cv2.bitwise_and(images[label][n], images[label][n], mask=mask)
            if sharpen is True:
                image = seg_image
                image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
                image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
                seg_image = image_sharp

            seg_images[label].append(seg_image)

    return seg_images


def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


images_segmented = segment_plant(images_per_class)
plot_raw_grid(images_segmented)



