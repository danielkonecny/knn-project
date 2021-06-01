# KNN Project - Traffic Sign Detector
# Authors: Daniel Konecny (xkonec75), Jan Pavlus (xpavlu10), David Sedlak (xsedla1d)

import torch as th
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torch.utils.data import Dataset, DataLoader

from classifier import load_dataset

import time
import tqdm

import argparse
import os

import pdb

import sys

sys.path.insert(0, '.')

import detector.classes


def loss_function(loss_input, target):
    _, labels = target.max(dim=1)
    return th.nn.CrossEntropyLoss()(loss_input, labels)


class Trainer(object):
    def __init__(self, model=models.resnet50(pretrained=True), device="cpu", number_of_classes=135):
        self.device = th.device(device)
        self.model = model
        self.model.fc = th.nn.Linear(2048, number_of_classes)
        self.model = self.model.double().to(self.device)
        self.curr_epoch = 0
        self.optimizer = th.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def train(self, train_dataloader):
        print("Training...")
        self.model.train()
        losses = list()
        for image, label in tqdm.tqdm(train_dataloader):
            self.optimizer.zero_grad()
            image_t = th.tensor(image, device=self.device, requires_grad=True)
            image_t = image_t.reshape(image_t.shape[0], image_t.shape[3], image_t.shape[1], image_t.shape[2])
            label_t = th.tensor(label, device=self.device)
            est = self.model(image_t.double())
            loss = loss_function(est, label_t)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        self.writer.add_scalar("Train", sum(losses) / float(len(losses)), self.curr_epoch)
        print("Epoch: {} Train: {}".format(self.curr_epoch, sum(losses) / float(len(losses))))

    def crossvalidate(self, cv_dataloader):
        print("Validating...")
        self.model.eval()
        with th.no_grad():
            losses = list()
            correct = 0
            counter = 0
            for image, label in tqdm.tqdm(cv_dataloader):
                image_t = th.tensor(image, device=self.device)
                image_t = image_t.reshape(image_t.shape[0], image_t.shape[3], image_t.shape[1], image_t.shape[2])
                label_t = th.tensor(label, device=self.device)
                est = self.model(image_t.double())
                loss = loss_function(est, label_t)
                losses.append(loss.item())
                correct += (est.argmax(dim=1) == label_t.argmax(dim=1)).float().sum().cpu()
                counter += image.shape[0]
            accuracy = 100 * (correct / counter)
            self.writer.add_scalar("Cross-validation", sum(losses) / float(len(losses)), self.curr_epoch)
            self.writer.add_scalar("Cross-validation accuracy", accuracy, self.curr_epoch)
            print(
                "Epoch: {} Cross-validation: {} Accuracy: {}".format(self.curr_epoch, sum(losses) / float(len(losses)),
                                                                     accuracy))

    def run(self, epochs, output_path, run_args):
        self.writer = SummaryWriter("{}/tensorboard".format(output_path))
        cv_dataloader = load_dataset.batch_provider(batch_size=run_args.batch, path=run_args.val_dataset,
                                                    split_name=run_args.split_name)
        self.crossvalidate(cv_dataloader)
        for self.curr_epoch in range(epochs):
            train_dataloader = load_dataset.batch_provider(batch_size=run_args.batch, path=run_args.train_dataset,
                                                           split_name=run_args.split_name)
            cv_dataloader = load_dataset.batch_provider(batch_size=run_args.batch, path=run_args.val_dataset,
                                                        split_name=run_args.split_name)
            print("Epoch: {} of {}".format(self.curr_epoch, epochs))
            self.train(train_dataloader)
            self.crossvalidate(cv_dataloader)
            th.save(self.model, "{}/model".format(output_path))


def classify(path, data, device):
    data_tensor = th.from_numpy(data).double()
    model = th.load(path, map_location=th.device(device))
    return model(data_tensor).detach().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dt", "--train_dataset",
        default="mapillary_numpyed",
        help="Path to folder with train dataset."
    )
    parser.add_argument(
        "-dv", "--val_dataset",
        default="mapillary_numpyed",
        help="Path to folder with validation dataset."
    )
    parser.add_argument(
        "-s", "--split_name",
        default="warning",
        help='["warning", "other-sign", "information", "regulatory", "complementary"]'
    )
    parser.add_argument(
        "-d", "--device",
        default="cpu",
        help='["cuda", "cpu"]'
    )
    parser.add_argument(
        "-o", "--output",
        default="../output1",
        help="Output folder"
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train on"
    )
    parser.add_argument(
        "-b", "--batch",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train on the dataset"
    )
    args = parser.parse_args()

    if args.train:
        trainer = Trainer(device=args.device, number_of_classes=len(detector.classes.splits_dict[args.split_name]))
        os.makedirs(args.output, exist_ok=True)
        trainer.run(epochs=args.epochs, output_path=args.output, run_args=args)
