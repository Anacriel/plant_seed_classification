import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from glob import glob
from mpl_toolkits.axes_grid1 import ImageGrid


def read_images(path):
    total = 0
    images = []
    titles = []

    for image_path in glob(os.path.join(path, "*.png")):
        titles.append(os.path.basename(image_path))
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        images.append(image_bgr)
        total = total + 1

    print(total)
    return images, titles


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
    fig = plt.figure(1, (8, 12))
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


def resize_images(path, height=200, width=200):
    images = []
    for image_path in glob(os.path.join(path, "*.png")):
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        resized_img = cv2.resize(image_bgr, (height, width))
        cv2.imwrite(image_path, resized_img)


def create_mask_for_plant(image):
    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def segment_plant(images, sharpen=True):                    # RW def_dict
    seg_images = []
    for n in range(len(images)):
        mask = create_mask_for_plant(images[n])
        seg_image = cv2.bitwise_and(images[n], images[n], mask=mask)
        if sharpen:
            image = seg_image
            image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
            image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
            seg_image = image_sharp

        seg_images.append(seg_image)

    return seg_images


def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


def find_contours(mask_image):
    return cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]


def calculate_largest_contour_area(contours):
    if len(contours) == 0:
        return 0
    c = max(contours, key=cv2.contourArea)
    return cv2.contourArea(c)


def calculate_contours_area(contours, min_contour_area=250):
    area = 0
    for c in contours:
        c_area = cv2.contourArea(c)
        if c_area >= min_contour_area:
            area += c_area
    return area


def main():
    data_dir = '../../data/raw/'
    proc_data_dir = '../../data/processed/'

    # Called only once, don't need any more...
    #resize_images(proc_data_dir + 'test')

    # Process resized pics (300x300px)
    images_per_class, titles = read_images(proc_data_dir + 'test')

    images_segmented = segment_plant(images_per_class)
    title = titles
    areas = []
    larges_contour_areas = []
    nb_of_contours = []
    images_height = []
    images_width = []

    for image in images_per_class:
        mask = create_mask_for_plant(image)
        contours = find_contours(mask)

        area = calculate_contours_area(contours)
        largest_area = calculate_largest_contour_area(contours)
        height, width, channels = image.shape

        images_height.append(height)
        images_width.append(width)
        areas.append(area)
        nb_of_contours.append(len(contours))
        larges_contour_areas.append(largest_area)

    features_df = pd.DataFrame()
    features_df["title"] = title
    features_df["area"] = areas
    features_df["largest_area"] = larges_contour_areas
    features_df["number_of_components"] = nb_of_contours
    features_df["height"] = images_height
    features_df["width"] = images_width

    features_df.to_csv("test_data.csv", sep='\t', encoding='utf-8')


if __name__ == '__main__':
    main()
