# KNN Project - Traffic Sign Detector
# Authors: Daniel Konecny (xkonec75), Jan Pavlus (xpavlu10), David Sedlak (xsedla1d)
import argparse

from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog

from detector.classes import grouped_classes_dict
from detector.dataset import preview_dataset, load_mapillary_dataset
from detector.model import define_model, train, predict, evaluate


def device_type(s: str) -> str:
    if s not in ["cpu", "gpu"]:
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
        "--device",
        default="cpu",
        type=device_type,
        help="Set to gpu for training on gpu."
    )
    parser.add_argument(
        "-m", "--model",
        default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        help="Specifies which model from detectron2 model zoo use, "
             "see 'https://github.com/facebookresearch/detectron2/blob/master"
             "/MODEL_ZOO.md'."
    )
    parser.add_argument(
        "-l", "--learning-rate",
        default=0.0125,
        type=float,
        help="Learning rate."
    )
    parser.add_argument(
        "-i", "--iterations",
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


def main(argv=None):
    setup_logger()
    args = parse_args(argv)

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


if __name__ == '__main__':
    exit(main())
