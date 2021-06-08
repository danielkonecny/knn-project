# KNN Project - Traffic Sign Detector
# Authors: Daniel Konecny (xkonec75), Jan Pavlus (xpavlu10), David Sedlak (xsedla1d)

import argparse
import glob
import sys

import numpy as np
import matplotlib.pyplot as plt

import detector.classes

sys.path.insert(0, '.')


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset",
        default="mapillary_numpyed",
        help="Path to folder with dataset."
    )
    parser.add_argument(
        "-D", "--dimensions",
        default="224,224,3",
        help="Dimensions of output images."
    )
    parser.add_argument(
        "-l", "--grouped_label",
        default="5",
        help="Number of labels when grouped."
    )
    parser.add_argument(
        "-L", "--detailed_label",
        default="401",
        help="Number of labels."
    )
    args = parser.parse_args(argv)
    return args


def parse_dimensions(dim_string):
    dimension_y = int(dim_string.split(',')[0])
    dimension_x = int(dim_string.split(',')[1])
    dimension_z = int(dim_string.split(',')[2])
    return dimension_y, dimension_x, dimension_z


def file_provider(path):
    image_files = glob.glob(f'{path}/images*.npy')
    image_files.sort()
    grouped_label_files = glob.glob(f'{path}/grouped_labels*.npy')
    grouped_label_files.sort()
    shuffle = np.random.permutation(len(image_files))
    for i in range(len(shuffle)):
        images = np.load(image_files[shuffle[i]])
        grouped_labels = np.load(grouped_label_files[shuffle[i]])
        yield images, grouped_labels


def batch_provider(batch_size, path, split_name, image_dimensions=(0, 3, 224, 224)):
    grouped_label_count = len(detector.classes.splits_dict[split_name])
    image_overflow = np.empty(image_dimensions)
    grouped_label_overflow = np.empty((0, grouped_label_count))

    for provided_images, provided_grouped_labels in file_provider(f"{path}/{split_name}"):
        images = np.concatenate((image_overflow, provided_images))
        grouped_labels = np.concatenate((grouped_label_overflow, provided_grouped_labels))

        shuffle = np.random.permutation(len(images))

        for i in range(0, len(shuffle), batch_size):
            if i + batch_size < len(shuffle):
                batch_indices = shuffle[i:i + batch_size]
                yield images[batch_indices], grouped_labels[batch_indices]
            else:
                overflow_indices = shuffle[i:]
                image_overflow = images[overflow_indices]
                grouped_label_overflow = grouped_labels[overflow_indices]
                break


def main(argv=None):
    args = parse_args(argv)
    dimension_y, dimension_x, dimension_z = parse_dimensions(args.dimensions)
    image_dimensions = (0, dimension_y, dimension_x, dimension_z)
    grouped_label_count = int(args.grouped_label)
    detailed_label_count = int(args.detailed_label)

    batch_size = 128

    for images, grouped_labels, detailed_labels in batch_provider(batch_size, args.dataset, image_dimensions,
                                                                  grouped_label_count, detailed_label_count):
        print(grouped_labels[0])
        print(detailed_labels[0])
        plt.imshow(images[0])
        plt.show()


if __name__ == '__main__':
    exit(main())
