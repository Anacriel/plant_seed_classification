# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import scipy.stats as st
import src.visualization.visualize as vs
import src.features.build_features as bf
from glob import glob


def get_images(input_path, height=200, width=200, per_class=0):
    images = []
    labels = []
    for class_dir_name in os.listdir(input_path):
        if '.DS_Store' == class_dir_name:
            continue
        class_dir_path = os.path.join(input_path, class_dir_name)
        cur_per_class = 0
        for image_path in glob(os.path.join(class_dir_path, "*.png")):
            image_bgr = cv2.imread(image_path)
            image_bgr = cv2.resize(image_bgr, (height, width), interpolation=cv2.INTER_AREA)
            images.append(image_bgr)
            labels.append(class_dir_name)
            cur_per_class = cur_per_class + 1

            if per_class and cur_per_class == per_class:
                break

    return images, labels


def create_dataset_added_features(images, labels, kind):
    features_names = [kind, 'area', 'largest_area', 'number_of_elems', 'perimeter',
                      'aspect_ratio', 'circularity', 'mean_r', 'mean_g', 'mean_b',
                      'stddev_r', 'stddev_g', 'stddev_b',
                      'skew_r', 'skew_g', 'skew_b']
    df = pd.DataFrame([], columns=features_names)
    for i in range(len(images)):
        # Delete background
        masked_img, mask = vs.segment_plant(images[i])
        img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
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
        #rectangularity = 0.0 if area == 0.0 else w * h / area
        circularity = 0.0 if area == 0.0 else (perimeter ** 2) / area

        # Color features
        red_channel = img[:, :, 0]
        green_channel = img[:, :, 1]
        blue_channel = img[:, :, 2]

        blue_channel_plant = blue_channel[blue_channel != 255]
        green_channel_plant = green_channel[green_channel != 255]
        red_channel_plant = red_channel[red_channel != 255]

        red_mean = np.mean(red_channel_plant)
        green_mean = np.mean(green_channel_plant)
        blue_mean = np.mean(blue_channel_plant)

        #red_med = np.median(red_channel_plant)
        #green_med = np.median(green_channel_plant)
        #blue_med = np.median(blue_channel_plant)

        red_std = np.std(red_channel_plant)
        green_std = np.std(green_channel_plant)
        blue_std = np.std(blue_channel_plant)

        tmp_r = np.reshape(red_channel, (-1))
        tmp_g = np.reshape(green_channel, (-1))
        tmp_b = np.reshape(blue_channel, (-1))

        skew_r = st.skew(tmp_r[tmp_r != 255])
        skew_g = st.skew(tmp_g[tmp_g != 255])
        skew_b = st.skew(tmp_b[tmp_b != 255])

        vector = [labels[i], area, largest_area, nb_of_contours, perimeter,
                  aspect_ratio,  circularity,
                  red_mean, green_mean, blue_mean, red_std, green_std, blue_std,
                  skew_r, skew_g, skew_b]

        df_temp = pd.DataFrame([vector], columns=features_names)
        df = df.append(df_temp)
    return df


def prepare_images(input_path, output_path, height=200, width=200):
    for class_dir_name in os.listdir(input_path):  # total images per class
        if '.DS_Store' != class_dir_name:
            class_dir_path = os.path.join(input_path, class_dir_name)
            for image_path in glob(os.path.join(class_dir_path, "*.png")):
                image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                resized_img = cv2.resize(image_bgr, (height, width))
                segmented_img = vs.segment_plant(resized_img)
                cv2.imwrite(output_path + image_path[len(input_path):], segmented_img)


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

    prepare_images(data_dir + 'train', proc_data_dir + 'train')
    prepare_images(data_dir + 'test', proc_data_dir + 'test')


if __name__ == '__main__':
    main()

