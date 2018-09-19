import argparse
import matplotlib
import torch
from torch import nn, optim
from PIL import Image
from classifier import img_classifier, fc_net, freeze_parameters, load_classifier


def process_img(img_path, mean, std):
    pass

def predict(image, classifier, device, topk):
    pass

def unprocess_img(image, mean, std):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument()
    args = parser.parse_args()


if (__name__ == '__main__'):
    main()
