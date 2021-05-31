# KNN Project - Traffic Sign Detector
# Authors: Daniel Konecny (xkonec75), Jan Pavlus (xpavlu10), David Sedlak (xsedla1d)
import argparse
import sys

import cv2
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

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
        raise argparse.ArgumentTypeError(f"Expecting either 'cpu' or 'gpu', got '{s}'.")
    return s


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset",
        default="mapillary",
        help="Path to folder with dataset."
    )
    parser.add_argument(
        "-i", "--image",
        default="mapillary/mtsd_v2_fully_annotated_images.val/images/zZyl1YiP_5xGXcDAprC6sQ.jpg",
        help="Path to input image."
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
        help="Exectuon mode:"
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
        default="detector_weights.pth",
        help="path to trained detector weights."
    )
    parser.add_argument(
        "--classifier-weights",
        default="classifier_final.pth",
        help="path to trained classifier weights."
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
        "--preview",
        action="store_true",
        help="Preview images after loading the dataset and show inference result "
             "after training.")

    args = parser.parse_args(argv)
    return args


def train_detector(args):
    # load dataset
    for d in ["train", "test", "val"]:
        DatasetCatalog.register(
            "traffic_signs_" + d,
            lambda d=d: load_mapillary_dataset(args.dataset, d)
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


def draw_output(im, instances):
    v = Visualizer(
        im[:, :, ::-1],
        scale=0.5,
        instance_mode=ColorMode.IMAGE  # remove the colors of unsegmented pixels
    )
    out = v.draw_instance_predictions(instances.to("cpu"))
    cv2.imshow('', out.get_image()[:, :, ::-1])
    cv2.waitKey(10000)


def detect(im, args, visuzalize=False):
    model = load_model(args.detector_weights, args.model, args.device)
    predictor = DefaultPredictor(model)
    outputs = predictor(im)

    if visuzalize:
        classes = outputs["instances"].get("pred_classes")
        class_names = []
        label_map = list(grouped_classes_dict.keys())
        for cls in classes:
            class_names.append(label_map[cls])
        outputs["instances"].set("pred_classes", class_names)
        draw_output(im, outputs["instances"])

    return outputs


def detect_and_classify(im, args, visualize=False):
    annotations = detect(im, args, visuzalize=False)
    cropped_all = crop_detected_signs(im, annotations)
    for cropped in cropped_all:
        classifier_output = classify(args.classifier_weights, cropped)
        print(classifier_output)

    if visualize:
        draw_output(im, annotations["instances"])


def main(argv=None):
    setup_logger()
    args = parse_args(argv)
    if args.exec == "training":
        train_detector(args)
    else:
        im = cv2.imread(args.image)
        if args.exec == "detection":
            detect(im, args, visuzalize=True)
        if args.exec == "detection&classification":
            detect_and_classify(im, args, visualize=True)


if __name__ == '__main__':
    exit(main())
