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


def get_all_samples(path,typeOfData):
    with open(f'{path}/mtsd_v2_fully_annotated_annotation/mtsd_v2_fully_annotated/splits/{typeOfData}.txt') as f:
        lines = [line.rstrip() for line in f]
    return lines


def load_image(path, filename, typeOfData):
    try:
        image = Image.open(f'{path}/mtsd_v2_fully_annotated_images.{typeOfData}/images/{filename}.jpg')
    except OSError:
        time.sleep(3)
        image = Image.open(f'{path}/mtsd_v2_fully_annotated_images.{typeOfData}/images/{filename}.jpg')

    try:
        with open(f'{path}/mtsd_v2_fully_annotated_annotation/mtsd_v2_fully_annotated/annotations/{filename}.json') as f:
            annotation = json.load(f)
    except OSError:
        time.sleep(3)
        with open(f'{path}/mtsd_v2_fully_annotated_annotation/mtsd_v2_fully_annotated/annotations/{filename}.json') as f:
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
    splitted_labels = np.zeros(len(annotation['objects']))

    for i, image_object in enumerate(annotation['objects']):
        group_label = image_object['label'].split('--')[0]
        image_label = detector.classes.grouped_classes_dict[group_label]
        grouped_labels[i] = image_label
        splitted_labels[i] = detector.classes.splits_dict[group_label][image_object['label']]

    return grouped_labels, splitted_labels


def get_detailed_labels_one_hot(annotation, label_count):
    labels = np.zeros((len(annotation['objects']), label_count))
    index = 0

    for image_object in annotation['objects']:
        image_label = detector.classes.classes_dict[image_object['label']]
        labels[index][image_label] = 1
        index += 1

    return labels


def get_detailed_labels(annotation, label_count):
    labels = np.zeros(len(annotation['objects']))

    for index, image_object in enumerate(annotation['objects']):
        image_label = detector.classes.classes_dict[image_object['label']]
        labels[index] = image_label

    return labels


# def get_splitted_labels(detailed, grouped, label_count):
#     labels = np.zeros(len(detailed))
#
#     for i, label_id in enumerate(detailed):
#         image_id = recompute_label(label_id, grouped[i])
#         labels[i]= image_id
#
#     return labels


def get_splitted_labels(annotation):
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
    grouped_label_count = int(args.grouped_label)
    detailed_label_count = int(args.detailed_label)
    split_names = list(detector.classes.grouped_classes_dict.keys())
    splits_samples = {}
    splits_labels = {}
    split_file_indexes = {}

    image_names = get_all_samples(args.dataset, args.type)

    for split_name in split_names:
        splits_samples[split_name] = np.empty((0, dimension_y, dimension_x, dimension_z))
        splits_labels[split_name] = np.empty((0, len(detector.classes.splits_dict[split_name])))
        split_file_indexes[split_name] = 0

    # samples = np.empty((0, dimension_y, dimension_x, dimension_z))
    # grouped_labels = np.empty((0, grouped_label_count))
    # detailed_labels = np.empty((0, detailed_label_count))
    #
    # file_index = 0
    image_count = len(image_names)

    for i, image_name in enumerate(image_names):
        print(f'Reading file {image_name}. Progress: {i}/{image_count}.')
        image, annotation = load_image(args.dataset, image_name, args.type)

        new_samples = extract_samples(image, annotation, dimension_y, dimension_x, dimension_z)
        new_grouped_labels, new_splitted_labels = get_labels(annotation)
        # new_detailed_labels = get_detailed_labels(annotation, detailed_label_count)
        # new_splitted_labels = get_splitted_labels(new_detailed_labels, new_grouped_labels, detailed_label_count)

        for split_name in split_names:
            split_mask = new_grouped_labels == detector.classes.grouped_classes_dict[split_name]
            splits_samples[split_name] = np.concatenate((splits_samples[split_name], new_samples[np.where(split_mask)]))
            splits_labels[split_name] = np.concatenate((splits_labels[split_name],
                                                        to_one_hot(new_splitted_labels[np.where(split_mask)],
                                                                   len(detector.classes.splits_dict[split_name]))))

            if len(splits_samples[split_name]) > 200:
                split_save_as_nd_array(args.output, split_name, splits_samples[split_name],
                                       splits_labels[split_name], split_file_indexes[split_name])
                split_file_indexes[split_name] += 1

                splits_samples[split_name] = np.empty(
                    (0, dimension_y, dimension_x, dimension_z))
                splits_labels[split_name] = np.empty(
                    (0, len(detector.classes.splits_dict[split_name])))

        # samples = np.concatenate((samples, new_samples))
        # grouped_labels = np.concatenate((grouped_labels, new_grouped_labels))
        # detailed_labels = np.concatenate((detailed_labels, new_detailed_labels))
        #
        # if len(samples) > 5000:
        #     save_as_nd_array(args.output, samples, grouped_labels, detailed_labels, file_index)
        #     file_index += 1
        #
        #     samples = np.empty((0, dimension_y, dimension_x, dimension_z))
        #     grouped_labels = np.empty((0, grouped_label_count))
        #     detailed_labels = np.empty((0, detailed_label_count))

    # save_as_nd_array(args.output, samples, grouped_labels, detailed_labels, file_index)

    for split_name in split_names:
        split_save_as_nd_array(args.output, split_name, splits_samples[split_name],
                               splits_labels[split_name],
                               split_file_indexes[split_name])


if __name__ == '__main__':
    exit(main())
