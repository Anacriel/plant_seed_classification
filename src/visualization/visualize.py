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


def plot_raw_grid(images, cols=12):
    fig = plt.figure(1, (8, 12))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(cols, cols),
                     axes_pad=0.05)
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


def create_mask_for_plant(image):
    #sensitivity = 35  # 35 is preferable
    #lower_hsv = np.array([60 - sensitivity, 100, 50])
    #upper_hsv = np.array([60 + sensitivity, 255, 255])

    ###
    blur = cv2.GaussianBlur(image, (5, 5), 2)
    image_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([25, 65, 0])
    upper_hsv = np.array([190, 255, 255])
    ###

    #image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def create_mask_for_plant_slic(image):
    sensitivity = 35  # 35 is preferable
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def segment_plant(images, data_type='train', sharpen=True):
    seg_images = images
    if data_type == 'train':
        seg_images = defaultdict(list)
        for label, val in images.items():
            for n in range(len(images[label])):
                mask = create_mask_for_plant_slic(images[label][n])
                seg_image = cv2.bitwise_and(images[label][n], images[label][n], mask=mask)
                if sharpen:
                    seg_image = sharpen_image(seg_image)

                # change background color to white
                seg_image[mask == 0] = 255
                seg_images[label].append(seg_image)

    elif data_type == 'test':
        seg_images = []
        for n in range(len(images)):
            mask = create_mask_for_plant_slic(images[n])
            seg_image = cv2.bitwise_and(images[n], images[n], mask=mask)
            if sharpen:
                seg_image = sharpen_image(seg_image)
            seg_image[mask == 0] = 255
            seg_images.append(seg_image)

    return seg_images


def segment_plant_slic(images, data_type='train', sharpen=True):
    seg_images = images
    if data_type == 'train':
        seg_images = defaultdict(list)
        for label, val in images.items():
            for n in range(len(images[label])):
                image_slic = seg.slic(images[label][n], n_segments=355)
                image_rgb = color.label2rgb(image_slic, images[label][n], kind='avg')
                image_rgb = image_rgb[:, :, ::-1]
                mask = create_mask_for_plant_slic(image_rgb)
                seg_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

                if sharpen:
                    seg_image = sharpen_image(seg_image)

                seg_image[mask == 0] = 255
                seg_image[mask != 0] = images[label][n][mask != 0]

                # change background color to white
                seg_images[label].append(seg_image)

    elif data_type == 'test':
        seg_images = []
        for n in range(len(images)):
            image_slic = seg.slic(images[n], n_segments=355)
            image_rgb = color.label2rgb(image_slic, images[n], kind='avg')
            image_rgb = image_rgb[:, :, ::-1]
            mask = create_mask_for_plant_slic(image_rgb)
            seg_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

            if sharpen:
                seg_image = sharpen_image(seg_image)

            # change background color to white
            seg_image[mask == 0] = 255
            seg_image[mask != 0] = images[n][mask != 0]

            seg_images.append(seg_image)

    return seg_images


def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

