# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import mahotas as mt
import src.visualization.visualize as vs
import src.features.build_features as bf
import matplotlib.pyplot as plt
from glob import glob


def get_images(input_path, height=200, width=200):
    images = []
    labels = []
    for class_dir_name in os.listdir(input_path):
        if '.DS_Store' == class_dir_name:
            continue
        class_dir_path = os.path.join(input_path, class_dir_name)
        for image_path in glob(os.path.join(class_dir_path, "*.png")):
            image_bgr = cv2.imread(image_path)
            image_bgr = cv2.resize(image_bgr, (height, width), interpolation=cv2.INTER_AREA)
            images.append(image_bgr)
            labels.append(class_dir_name)

    return images, labels


def create_dataset(images, labels):
    features_names = ['label', 'area', 'largest_area', 'number_of_elems', 'perimeter', 'physiological_length', \
                      'physiological_width', 'aspect_ratio', 'rectangularity', 'circularity', \
                      'mean_r', 'mean_g', 'mean_b', 'stddev_r', 'stddev_g', 'stddev_b', \
                      'contrast', 'correlation', 'entropy'
                      ]
    df = pd.DataFrame([], columns=features_names)
    for i in range(len(images)):
        # Reduce noise
        images[i] = cv2.fastNlMeansDenoisingColored(images[i], None, 5, 5, 5, 5)

        # Delete background
        mask = vs.create_mask_for_plant(images[i])

        seg_image = cv2.bitwise_and(images[i], images[i], mask=mask)
        seg_image = vs.sharpen_image(seg_image)
        seg_image[mask == 0] = 255

        # Preprocessing 1
        img = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
        gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #ret_otsu, im_bw_otsu = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #kernel = np.ones((5, 5), np.uint8)
        #closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

        # Preprocessing 2


        # Shape features
        #contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = bf.find_contours(mask)
        if not contours:
            continue

        nb_of_contours = len(contours)
        cnt = contours[0]
        M = cv2.moments(cnt)
        #area = cv2.contourArea(cnt)
        area = bf.calculate_contours_area(contours)
        largest_area = bf.calculate_largest_contour_area(contours)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = 0.0 if h == 0.0 else float(w) / h
        rectangularity = 0.0 if area == 0.0 else w * h / area
        circularity = 0.0 if area == 0.0 else (perimeter ** 2) / area

        # Color features
        red_channel = img[:, :, 0]
        green_channel = img[:, :, 1]
        blue_channel = img[:, :, 2]

        blue_channel[blue_channel == 255] = 0
        green_channel[green_channel == 255] = 0
        red_channel[red_channel == 255] = 0

        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)

        red_std = np.std(red_channel)
        green_std = np.std(green_channel)
        blue_std = np.std(blue_channel)

        # Texture features
        textures = mt.features.haralick(gs)
        ht_mean = textures.mean(axis=0)
        contrast = ht_mean[1]
        correlation = ht_mean[2]
        entropy = ht_mean[8]

        vector = [labels[i], area, largest_area, nb_of_contours, perimeter, w, h,
                  aspect_ratio, rectangularity, circularity,
                  red_mean, green_mean, blue_mean, red_std, green_std, blue_std,
                  contrast, correlation, entropy]

        df_temp = pd.DataFrame([vector], columns=features_names)
        df = df.append(df_temp)
    return df


def create_dataset_added_features(images, labels, kind, to_dataframe=True):
    features_names = [kind, 'area', 'largest_area', 'number_of_elems', 'perimeter', 'physiological_length',
                      'physiological_width', 'aspect_ratio', 'rectangularity', 'circularity',
                      'mean_r', 'mean_g', 'mean_b', 'stddev_r', 'stddev_g', 'stddev_b',
                      'contrast', 'correlation', 'entropy'
                      ]
    df = pd.DataFrame([], columns=features_names)
    for i in range(len(images)):
        # Reduce noise
        images[i] = cv2.fastNlMeansDenoisingColored(images[i], None, 5, 5, 5, 5)

        # Delete background (prev)
        #mask = vs.create_mask_for_plant(images[i])
        #seg_image = cv2.bitwise_and(images[i], images[i], mask=mask)
        #seg_image = vs.sharpen_image(seg_image)
        #seg_image[mask == 0] = 255

        blur = cv2.GaussianBlur(images[i], (3, 3), 2)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lower_green = np.array([22, 60, 0])  # second 75
        upper_green = np.array([150, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masked_img = cv2.bitwise_and(images[i], images[i], mask=opened_mask)

        masked_img[mask == 0] = 255

        kernel = np.ones((2, 2), np.uint8)
        dilation = cv2.dilate(masked_img, kernel, iterations=1)

        img = cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB)
        gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        contours = bf.find_contours(mask)
        if not contours:
            continue

        nb_of_contours = bf.count_contours(contours)
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        area = bf.calculate_contours_area(contours)
        largest_area = bf.calculate_largest_contour_area(contours)
        perimeter = cv2.arcLength(cnt, True)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = 0.0 if h == 0.0 else float(w) / h
        rectangularity = 0.0 if area == 0.0 else w * h / area
        circularity = 0.0 if area == 0.0 else (perimeter ** 2) / area

        # Color features
        red_channel = img[:, :, 0]
        green_channel = img[:, :, 1]
        blue_channel = img[:, :, 2]

        blue_channel[blue_channel == 255] = 0
        green_channel[green_channel == 255] = 0
        red_channel[red_channel == 255] = 0

        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)

        red_std = np.std(red_channel)
        green_std = np.std(green_channel)
        blue_std = np.std(blue_channel)

        # Texture features
        textures = mt.features.haralick(gs)
        ht_mean = textures.mean(axis=0)
        contrast = ht_mean[1]
        correlation = ht_mean[2]
        entropy = ht_mean[8]

        vector = [labels[i], area, largest_area, nb_of_contours, perimeter, w, h,
                  aspect_ratio, rectangularity, circularity,
                  red_mean, green_mean, blue_mean, red_std, green_std, blue_std,
                  contrast, correlation, entropy]

        df_temp = pd.DataFrame([vector], columns=features_names)
        df = df.append(df_temp)
    return df


def resize_images(input_path, output_path, height=300, width=300):
    for class_dir_name in os.listdir(input_path):  # total images per class
        if '.DS_Store' != class_dir_name:
            class_dir_path = os.path.join(input_path, class_dir_name)
            for image_path in glob(os.path.join(class_dir_path, "*.png")):
                image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                resized_img = cv2.resize(image_bgr, (height, width))
                cv2.imwrite(output_path + image_path[len(input_path):], resized_img)


def increase_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    data_dir = '../../data/raw/'
    proc_data_dir = '../../data/processed/'

    resize_images(data_dir + 'train', proc_data_dir + 'train')
    resize_images(data_dir + 'test', proc_data_dir + 'test')


if __name__ == '__main__':
    main()

