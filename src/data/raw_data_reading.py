import matplotlib.pyplot as plt
import os
import cv2
from glob import glob


data_dir = '../../data/raw/'

test_image_dir = os.path.join(data_dir, 'test')
train_image_dir = os.path.join(data_dir, 'train')

# submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
# print(submission.head())  # testing if read correctly


def read_images(path):
    images = {}
    for class_dir_name in os.listdir(path):  # total images per class
        if '.DS_Store' != class_dir_name:
            class_dir_path = os.path.join(path, class_dir_name)
            class_label = class_dir_name

            images[class_label] = []
            for image_path in glob(os.path.join(class_dir_path, "*.png")):
                image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                images[class_label].append(image_bgr)

    return images


images_per_class = read_images(train_image_dir)


for key, value in images_per_class.items():
    print("{0} -> {1}".format(key, len(value)))


def plot_class(label, images, rows=3, cols=3):
    fig, axs = plt.subplots(rows, cols, figsize=(6, 6))
    n = 0
    for i in range(0, rows):
        for j in range(0, cols):
            axs[i, j].xaxis.set_ticklabels([])
            axs[i, j].yaxis.set_ticklabels([])
            axs[i, j].imshow(images[label][n])
            n += 1

    plt.suptitle(label)
    plt.show()


for key, value in images_per_class.items():
    plot_class(key, images_per_class)

