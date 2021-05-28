# KNN Project - Traffic Sign Detector
# Authors: Daniel Konecny (xkonec75), Jan Pavlus (xpavlu10), David Sedlak (xsedla1d)

import json
import math
import cv2
import random
import os
import numpy as np

from PIL import Image
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

from detector.classes import grouped_classes_dict


def load_mapillary_dataset(path, split="train"):
    annotations_folder = path+"/mtsd_v2_fully_annotated_annotation" \
                              "/mtsd_v2_fully_annotated/"
    train_images_folder = path+"/mtsd_v2_fully_annotated_images." + split + "/images/"
    images = set([os.path.splitext(f)[0] for f in os.listdir(train_images_folder)
                  if os.path.isfile(os.path.join(train_images_folder, f))])

    with open(annotations_folder + "/splits/" + split + ".txt") as f:
        train_data = [i.rstrip("\n ") for i in f.readlines()]

    dataset_dicts = []
    for filename in train_data:
        if filename in images:
            try:
                with open(annotations_folder + "/annotations/" + filename + ".json") as f:
                    annotation = json.load(f)

                    annotations = []
                    for obj in annotation['objects']:
                        annot = {
                            'bbox' : [obj['bbox']['xmin'],
                                      obj['bbox']['ymin'],
                                      obj['bbox']['xmax'],
                                      obj['bbox']['ymax']],
                            'bbox_mode' : BoxMode.XYXY_ABS,
                            'category_id': grouped_classes_dict[obj['label'].split('--')[0]]
                        }
                        annotations.append(annot)
                    image_info = {
                        'image_id': filename,
                        'file_name': os.path.join(train_images_folder, filename + ".jpg"),
                        'height': annotation['height'],
                        'width': annotation['width'],
                        'annotations': annotations
                    }
                    dataset_dicts.append(image_info)
            except FileNotFoundError:
                print(f"Annotation file for {filename} not found.")

    return dataset_dicts


def preview_dataset(dataset, path):
    dataset_dicts = load_mapillary_dataset(path, "train")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(dataset),
                                scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow('', out.get_image()[:, :, ::-1])
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def crop_detected_signs(im, annotations, dimension_y, dimension_x):
    pimage = Image.fromarray(im)
    croped_all = []

    for box in annotations["instances"].get_fields()["pred_boxes"]:
        x_min = math.floor(box[0])
        y_min = math.floor(box[1])
        x_max = math.ceil(box[2])
        y_max = math.ceil(box[3])

        cropped = pimage.crop((x_min, y_min, x_max, y_max))
        resized = cropped.resize((dimension_y, dimension_x), Image.ANTIALIAS)
        numpyed = np.array(resized)
        numpyed = numpyed.astype('float32') / 255.0
        croped_all.append(numpyed)

    return np.array(croped_all)
