import torch as th
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torch.utils.data import Dataset, DataLoader

import load_dataset

import time
import tqdm

import argparse
import os

import pdb

import sys
sys.path.insert(0, '.')

import detector.classes

class Trainer(object):
    def __init__(self, model = models.resnet50(pretrained=True), device = "cpu", numberOfClasses = 135):
        self.device = th.device(device)
        self.model = model
        self.model.fc = th.nn.Linear(2048, numberOfClasses)
        self.model = self.model.double().to(self.device)
        self.curr_epoch = 0
        self.optimizer = th.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    def loss_function(self, input, target):
        _, labels = target.max(dim=1)
        return th.nn.CrossEntropyLoss()(input, labels)

    def train(self, train_dataloader):
        print("Training...")
        self.model.train()
        losses = list()
        for image, label in tqdm.tqdm(train_dataloader):
            self.optimizer.zero_grad()
            imageT = th.tensor(image, device=self.device, requires_grad=True)
            imageT = imageT.reshape(imageT.shape[0], imageT.shape[3], imageT.shape[1], imageT.shape[2])
            labelT = th.tensor(label, device=self.device)
            est = self.model(imageT.double())
            loss = self.loss_function(est, labelT)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        self.writer.add_scalar("Train", sum(losses)/float(len(losses)), self.curr_epoch)
        print("Epoch: {} Train: {}".format(self.curr_epoch, sum(losses)/float(len(losses))))


    def crossvalidate(self, cv_dataloader):
        print("Validating...")
        self.model.eval()
        with th.no_grad():
            losses = list()
            correct = 0
            counter = 0
            for image, label in tqdm.tqdm(cv_dataloader):
                imageT = th.tensor(image, device=self.device)
                imageT = imageT.reshape(imageT.shape[0], imageT.shape[3], imageT.shape[1], imageT.shape[2])
                labelT = th.tensor(label, device=self.device)
                est = self.model(imageT.double())
                loss = self.loss_function(est, labelT)
                losses.append(loss.item())
                correct += (est.argmax(dim=1) == labelT.argmax(dim=1)).float().sum().cpu()
                counter += image.shape[0]
            accuracy = 100 * (correct / counter)
            self.writer.add_scalar("Cross-validation", sum(losses)/float(len(losses)), self.curr_epoch)
            self.writer.add_scalar("Cross-validation accuracy", accuracy, self.curr_epoch)
            print("Epoch: {} Cross-validation: {} Accurancy: {}".format(self.curr_epoch, sum(losses)/float(len(losses)), accuracy))

    def run(self, epochs, output_path, args):
        self.writer = SummaryWriter("{}/tensorboard".format(output_path))
        cv_dataloader = load_dataset.batch_provider(batch_size = args.batch, path = args.val_dataset, split_name=args.split_name)
        self.crossvalidate(cv_dataloader)
        for self.curr_epoch in range(epochs):
            train_dataloader = load_dataset.batch_provider(batch_size = args.batch, path = args.train_dataset, split_name=args.split_name)
            cv_dataloader = load_dataset.batch_provider(batch_size = args.batch, path = args.val_dataset, split_name=args.split_name)
            print("Epoch: {} of {}".format(self.curr_epoch,epochs))
            self.train(train_dataloader)
            self.crossvalidate(cv_dataloader)
            th.save(self.model, "{}/model".format(output_path))

def classify(path, data):
    model = th.load(path)
    return model(data)

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
        type = int,
        default = 100,
        help="Number of epochs to train on"
    )
    parser.add_argument(
        "-b", "--batch",
        type = int,
        default = 32,
        help="Batch size"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train on the dataset")
    args = parser.parse_args()

    if args.train:
        trainer = Trainer(device = args.device, numberOfClasses = len(detector.classes.splits_dict[args.split_name]))
        os.makedirs(args.output, exist_ok=True)
        trainer.run(epochs = args.epochs, output_path =  args.output, args = args)