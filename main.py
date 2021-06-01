# KNN Project - Traffic Sign Detector
# Authors: Daniel Konecny (xkonec75), Jan Pavlus (xpavlu10), David Sedlak (xsedla1d)
import argparse
import datetime
import os
import sys

import numpy as np

import cv2
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

import detector.classes
from classifier.classifier import classify
from detector.classes import grouped_classes_dict
from detector.dataset import preview_dataset, load_mapillary_dataset, crop_detected_signs
from detector.model import define_model, train, predict, evaluate, load_model

sys.path.insert(0, '.')


def device_type(s: str) -> str:
    if s not in ["cpu", "gpu"]:
        raise argparse.ArgumentTypeError(f"Expecting either 'cpu' or 'gpu', got '{s}'.")
    return s


def exec_mode(s: str) -> str:
    if s not in ["detection&classification", "detection", "training"]:
        raise argparse.ArgumentTypeError(
            f"Expecting either 'detection&classification', 'detection' or 'training', got '{s}'.")
    return s


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset",
        default="mapillary",
        help="Path to folder with dataset."
    )
    parser.add_argument(
        "-i", "--input",
        default="mapillary/mtsd_v2_fully_annotated_images.val/images/zZyl1YiP_5xGXcDAprC6sQ.jpg",
        help="Path to input image, or folder with images."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=device_type,
        help="Set to gpu for training on gpu."
    )
    parser.add_argument(
        "-e", "--exec",
        default="detection&classification",
        type=exec_mode,
        help="Execution mode:"
             "detection&classification (default): Run detection and classification."
             "detection: Run detection only."
             "training: Run detector training."
    )
    parser.add_argument(
        "-m", "--model",
        default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        help="Specifies which model from detectron2 model zoo use, "
             "see 'https://github.com/facebookresearch/detectron2/blob/master"
             "/MODEL_ZOO.md'."
    )
    parser.add_argument(
        "--detector-weights",
        default="models/detector.pth",
        help="Path to trained detector."
    )
    parser.add_argument(
        "--classifier-weights",
        default="models/classifiers",
        help="Path to folder with trained classifiers."
    )
    parser.add_argument(
        "-l", "--learning-rate",
        default=0.0125,
        type=float,
        help="Learning rate."
    )
    parser.add_argument(
        "--iterations",
        default=1000,
        type=int,
        help="Number of iterations to use."
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        help="Batch size."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Preview images after loading the dataset and show output results."
             "after training.")

    args = parser.parse_args(argv)
    return args


def train_detector(args):
    # load dataset
    for d in ["train", "test", "val"]:
        DatasetCatalog.register(
            "traffic_signs_" + d,
            lambda d2=d: load_mapillary_dataset(args.dataset, d2)
        )
        MetadataCatalog.get("traffic_signs_" + d).set(
            thing_classes=list(grouped_classes_dict.keys())
        )

    if args.preview:
        preview_dataset(args.dataset, "traffic_signs_train")

    # define model
    training_model = define_model(
        train_dts="traffic_signs_train",
        val_dts="traffic_signs_val",
        device=args.device,
        model=args.model,
        lr=args.learning_rate,
        iterations=args.iterations,
        batch_size=args.batch_size
    )

    trainer = train(training_model)

    if args.preview:
        predict(training_model, args.dataset, "traffic_signs_train")

    evaluate(training_model, trainer)


def draw_output(im, instances, output_dir, filename, visualize=False):
    v = Visualizer(
        im[:, :, ::-1],
        scale=0.5,
        instance_mode=ColorMode.IMAGE  # remove the colors of unsegmented pixels
    )
    out = v.draw_instance_predictions(instances.to("cpu"))
    cv2.imwrite(f"{output_dir}/{filename}", out.get_image()[:, :, ::-1])
    if visualize:
        cv2.imshow('', out.get_image()[:, :, ::-1])
        cv2.waitKey(10000)


def detect(im, args, out_dir, filename, visualize=False):
    model = load_model(args.detector_weights, args.model, args.device)
    predictor = DefaultPredictor(model)
    outputs = predictor(im)

    classes = outputs["instances"].get("pred_classes")
    class_names = []
    label_map = list(grouped_classes_dict.keys())
    for cls in classes:
        class_names.append(label_map[cls])
    outputs["instances"].set("pred_classes", class_names)
    draw_output(im, outputs["instances"], out_dir, filename, visualize)

    return outputs


def get_classification(predicted_class, predictions):
    prediction_index = np.argmax(predictions)
    sign_class = ""
    if predicted_class == "warning":
        sign_class = list(detector.classes.splitted_warning_dict.keys())[
            list(detector.classes.splitted_warning_dict.values()).index(prediction_index)]
    elif predicted_class == "information":
        sign_class = list(detector.classes.splitted_information_dict.keys())[
            list(detector.classes.splitted_information_dict.values()).index(prediction_index)]
    elif predicted_class == "regulatory":
        sign_class = list(detector.classes.splitted_regulatory_dict.keys())[
            list(detector.classes.splitted_regulatory_dict.values()).index(prediction_index)]
    elif predicted_class == "complementary":
        sign_class = list(detector.classes.splitted_complementary_dict.keys())[
            list(detector.classes.splitted_complementary_dict.values()).index(prediction_index)]

    return sign_class


def detect_and_classify(im, args, out_dir, filename, visualize=False):
    annotations = detect(im, args, out_dir, filename, visualize=False)
    try:
        classes = annotations["instances"].get_fields()["pred_classes"]
        boxes = annotations["instances"].get_fields()["pred_boxes"]
    except IndexError:
        # no annotations
        return

    cropped_all = crop_detected_signs(im, annotations, 224, 224)
    for cropped, predicted_class, box in zip(cropped_all, classes, boxes):
        cropped = cropped.reshape((1, cropped.shape[2], cropped.shape[0], cropped.shape[1]))
        if predicted_class == 'other-sign':
            print(f"Box: {box} - Class: {predicted_class}")
        elif predicted_class == 'regulatory':
            print(f"Box: {box} - Class: {predicted_class}")
        elif predicted_class == 'complementary':
            print(f"Box: {box} - Class: {predicted_class}")
        else:
            model_path = args.classifier_weights + "/" + predicted_class
            classifier_output = classify(model_path, cropped, args.device)
            sign_class = get_classification(predicted_class, classifier_output)
            print(f"Box: {box} - Class: {sign_class}")

    draw_output(im, annotations["instances"], out_dir, filename, visualize)


def execute_on_image(args, filename, im, out_dir):
    if args.exec == "detection":
        detect(im, args, out_dir, filename, visualize=args.visualize)
    if args.exec == "detection&classification":
        detect_and_classify(im, args, out_dir, filename, visualize=args.visualize)


def main(argv=None):
    setup_logger()
    args = parse_args(argv)
    if args.exec == "training":
        train_detector(args)
    else:
        out_dir = f"output-{args.exec}-" + datetime.datetime.now().strftime("%d-%m-%Y-(%H:%M:%S)")
        os.mkdir(out_dir)
        if os.path.isdir(args.input):
            for filename in os.listdir(args.input):
                pth = f"{args.input}/{filename}"
                if os.path.isfile(pth):
                    print(pth)
                    execute_on_image(args, os.path.basename(filename), cv2.imread(pth), out_dir)
        else:
            execute_on_image(args, os.path.basename(args.input), cv2.imread(args.input), out_dir)


if __name__ == '__main__':
    exit(main())
