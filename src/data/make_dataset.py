# -*- coding: utf-8 -*-
import os
import cv2
from glob import glob


def get_images(input_path, height=300, width=300):
    images = []
    labels = []
    for class_dir_name in os.listdir(input_path):
        if '.DS_Store' == class_dir_name:
            continue
        class_dir_path = os.path.join(input_path, class_dir_name)
        for image_path in glob(os.path.join(class_dir_path, "*.png")):
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_bgr = cv2.resize(image_bgr, (height, width))
            images.append(image_bgr)
            labels.append(class_dir_name)

    return images, labels


def resize_images(input_path, output_path, height=300, width=300):
    for class_dir_name in os.listdir(input_path):  # total images per class
        if '.DS_Store' != class_dir_name:
            class_dir_path = os.path.join(input_path, class_dir_name)
            for image_path in glob(os.path.join(class_dir_path, "*.png")):
                image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                resized_img = cv2.resize(image_bgr, (height, width))
                cv2.imwrite(output_path + image_path[len(input_path):], resized_img)


def delete_background(input_path, output_path):
    for class_dir_name in os.listdir(input_path):  # total images per class
        if '.DS_Store' != class_dir_name:
            class_dir_path = os.path.join(input_path, class_dir_name)
            for image_path in glob(os.path.join(class_dir_path, "*.png")):
                image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                cv2.imwrite(output_path + image_path[len(input_path):], image_bgr)


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

