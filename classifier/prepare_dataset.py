# KNN Project - Traffic Sign Detector
# Authors: Daniel Konecny (xkonec75), Jan Pavlus (xpavlu10), David Sedlak (xsedla1d)

import argparse
import json
from PIL import Image
import math
import numpy as np
from pathlib import Path
import time
import sys
import torch as th

sys.path.insert(0, '.')

import detector.classes


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset",
        default="mapillary",
        help="Path to folder with dataset."
    )
    parser.add_argument(
        "-D", "--dimensions",
        default="224,224,3",
        help="Dimensions of output images."
    )
    parser.add_argument(
        "-o", "--output",
        default="mapillary_numpyed",
        help="Path to folder where transformed dataset is saved."
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
    parser.add_argument(
        "-T", "--type",
        default="train",
        help="train/val/test"
    )
    args = parser.parse_args(argv)
    return args


def parse_dimensions(dim_string):
    dimension_y = int(dim_string.split(',')[0])
    dimension_x = int(dim_string.split(',')[1])
    dimension_z = int(dim_string.split(',')[2])
    return dimension_y, dimension_x, dimension_z


def get_all_samples(path, type_of_data):
    with open(f'{path}/mtsd_v2_fully_annotated_annotation/mtsd_v2_fully_annotated/splits/{type_of_data}.txt') as f:
        lines = [line.rstrip() for line in f]
    return lines


def load_image(path, filename, type_of_data):
    try:
        image = Image.open(f'{path}/mtsd_v2_fully_annotated_images.{type_of_data}/images/{filename}.jpg')
    except OSError:
        time.sleep(3)
        image = Image.open(f'{path}/mtsd_v2_fully_annotated_images.{type_of_data}/images/{filename}.jpg')

    try:
        with open(
                f'{path}/mtsd_v2_fully_annotated_annotation/mtsd_v2_fully_annotated/annotations/{filename}.json') as f:
            annotation = json.load(f)
    except OSError:
        time.sleep(3)
        with open(
                f'{path}/mtsd_v2_fully_annotated_annotation/mtsd_v2_fully_annotated/annotations/{filename}.json') as f:
            annotation = json.load(f)

    return image, annotation


def extract_samples(image, annotation, dimension_y, dimension_x, dimension_z):
    samples = np.empty((len(annotation['objects']), dimension_y, dimension_x, dimension_z))
    index = 0

    for image_object in annotation['objects']:
        x_min = math.floor(image_object['bbox']['xmin'])
        y_min = math.floor(image_object['bbox']['ymin'])
        x_max = math.ceil(image_object['bbox']['xmax'])
        y_max = math.ceil(image_object['bbox']['ymax'])

        cropped = image.crop((x_min, y_min, x_max, y_max))
        resized = cropped.resize((dimension_y, dimension_x), Image.ANTIALIAS)
        numpyed = np.array(resized)
        numpyed = numpyed.astype('float32') / 255.0
        samples[index] = numpyed

        index += 1

    return samples


def get_grouped_labels_one_hot(annotation, label_count):
    labels = np.zeros((len(annotation['objects']), label_count))
    index = 0

    for image_object in annotation['objects']:
        group_label = image_object['label'].split('--')[0]
        image_label = detector.classes.grouped_classes_dict[group_label]
        labels[index][image_label] = 1
        index += 1

    return labels


def get_labels(annotation):
    grouped_labels = np.zeros(len(annotation['objects']))
    split_labels = np.zeros(len(annotation['objects']))

    for i, image_object in enumerate(annotation['objects']):
        group_label = image_object['label'].split('--')[0]
        image_label = detector.classes.grouped_classes_dict[group_label]
        grouped_labels[i] = image_label
        split_labels[i] = detector.classes.splits_dict[group_label][image_object['label']]

    return grouped_labels, split_labels


def get_detailed_labels_one_hot(annotation, label_count):
    labels = np.zeros((len(annotation['objects']), label_count))
    index = 0

    for image_object in annotation['objects']:
        image_label = detector.classes.classes_dict[image_object['label']]
        labels[index][image_label] = 1
        index += 1

    return labels


def get_detailed_labels(annotation):
    labels = np.zeros(len(annotation['objects']))

    for index, image_object in enumerate(annotation['objects']):
        image_label = detector.classes.classes_dict[image_object['label']]
        labels[index] = image_label

    return labels


def get_split_labels(annotation):
    labels = np.zeros(len(annotation['objects']))

    for i, image_object in enumerate(annotation['objects']):
        group_label = image_object['label'].split('--')[0]
        labels[i] = detector.classes.splits_dict[group_label][image_object['label']]

    return labels


def to_one_hot(values, vector_size):
    result = np.zeros((len(values), vector_size))

    for i, value in enumerate(values):
        result[i][int(value)] = 1

    return result


def recompute_label(label_id, split_id):
    label_name = list(detector.classes.classes_dict.keys())[int(label_id)]
    split_name = list(detector.classes.grouped_classes_dict.keys())[int(split_id)]
    return detector.classes.splits_dict[split_name][label_name]


def save_as_nd_array(path, samples, grouped_labels, detailed_labels, file_index):
    Path(f'{path}').mkdir(parents=True, exist_ok=True)

    try:
        np.save(f'{path}/images{file_index:03d}.npy', samples)
    except OSError:
        time.sleep(3)
        np.save(f'{path}/images{file_index:03d}.npy', samples)
    print(f'Samples saved with shape {samples.shape}.')

    try:
        np.save(f'{path}/grouped_labels{file_index:03d}.npy', grouped_labels)
    except OSError:
        time.sleep(3)
        np.save(f'{path}/grouped_labels{file_index:03d}.npy', grouped_labels)
    print(f'Group labels saved with shape {grouped_labels.shape}.')

    try:
        np.save(f'{path}/detailed_labels{file_index:03d}.npy', detailed_labels)
    except OSError:
        time.sleep(3)
        np.save(f'{path}/detailed_labels{file_index:03d}.npy', detailed_labels)
    print(f'Detailed labels saved with shape {detailed_labels.shape}.')


def split_save_as_nd_array(path, split_name, samples, labels, file_index):
    Path(f'{path}/{split_name}').mkdir(parents=True, exist_ok=True)
    samples = samples.reshape((samples.shape[0], samples.shape[3], samples.shape[1], samples.shape[2]))

    try:
        np.save(f'{path}/{split_name}/images{file_index:03d}.npy', samples)
    except OSError:
        time.sleep(3)
        np.save(f'{path}/{split_name}/images{file_index:03d}.npy', samples)
    print(f'Samples saved with shape {samples.shape}.')

    try:
        np.save(f'{path}/{split_name}/grouped_labels{file_index:03d}.npy', labels)
    except OSError:
        time.sleep(3)
        np.save(f'{path}/{split_name}/grouped_labels{file_index:03d}.npy', labels)
    print(f'Group labels saved with shape {labels.shape}.')


def main(argv=None):
    args = parse_args(argv)
    dimension_y, dimension_x, dimension_z = parse_dimensions(args.dimensions)
    split_names = list(detector.classes.grouped_classes_dict.keys())
    splits_samples = {}
    splits_labels = {}
    split_file_indexes = {}
    splits_counts = {}

    image_names = get_all_samples(args.dataset, args.type)

    for split_name in split_names:
        splits_samples[split_name] = []
        splits_labels[split_name] = []
        splits_counts[split_name] = 0
        split_file_indexes[split_name] = 0

    image_count = len(image_names)

    for i, image_name in enumerate(image_names):
        print(f'Reading file {image_name}. Progress: {i}/{image_count}.')
        image, annotation = load_image(args.dataset, image_name, args.type)

        new_samples = extract_samples(image, annotation, dimension_y, dimension_x, dimension_z)
        new_grouped_labels, new_split_labels = get_labels(annotation)

        for split_name in split_names:
            split_mask = new_grouped_labels == detector.classes.grouped_classes_dict[split_name]
            samples = new_samples[np.where(split_mask)]
            splits_counts[split_name] += len(samples)
            splits_samples[split_name].append(samples)
            splits_labels[split_name].append(
                to_one_hot(new_split_labels[np.where(split_mask)], len(detector.classes.splits_dict[split_name])))

            if splits_counts[split_name] > 200:
                split_save_as_nd_array(args.output, split_name, np.concatenate(splits_samples[split_name]),
                                       np.concatenate(splits_labels[split_name]), split_file_indexes[split_name])
                split_file_indexes[split_name] += 1

                splits_samples[split_name] = []
                splits_labels[split_name] = []
                splits_counts[split_name] = 0

    for split_name in split_names:
        split_save_as_nd_array(args.output, split_name, splits_samples[split_name],
                               splits_labels[split_name],
                               split_file_indexes[split_name])


if __name__ == '__main__':
    exit(main())
