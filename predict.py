import argparse
import matplotlib
import numpy as np
import torch
import json
from torch import nn, optim
from PIL import Image
from classifier import img_classifier, fc_net, freeze_parameters, load_classifier


def process_img(img_path, mean, std):
    '''
    '''
    image = Image.open(img_path)

    # Scaling
    width, height = image.size
    scale = 256 / min(width, height)
    new_width, new_height = int(width * scale), int(height * scale)
    image = image.resize((new_width, new_height))

    # Cropping
    left, right = new_width // 2 - 112, new_width // 2 + 112
    upper, lower = new_height // 2 - 112, new_height // 2 + 112
    image = image.crop((left, upper, right, lower))

    # Normalizing and reordering
    image, mean, std = np.array(image), np.array(mean), np.array(std)
    image = ((image / 255 - mean) / std).transpose(2, 0, 1)

    return image


def unprocess_img(image, mean, std):
    '''
    '''
    mean, std = np.array(mean), np.array(std)
    image = ((image.transpose(1, 2, 0)) * std + mean).clip(0, 1)

    return image


def predict(image, classifier, device, topk):
    '''
    Takes in a processed image as a numpy array and makes a prediction
    Arguments
    ---------
    image           np.array
    classifier      nn.Module
    device          str
    topk            int
    '''
    image = torch.from_numpy(image).float().unsqueeze_(0)
    classifier.model.eval()
    classifier.model.to(device)
    image.to(device)

    probs, idxs = torch.exp(classifier.model(image)).topk(topk)
    probs = probs.detach().numpy().ravel()
    idxs = idxs.detach().numpy().ravel()
    idx_to_cat = {i : int(c) for c, i in classifier.cat_to_idx.items()}
    cats = np.vectorize(idx_to_cat.get)(idxs)

    probs = probs.tolist()
    cats = [str(cat) for cat in cats.tolist()]

    return probs, cats


def show_prediction(img_path, probs, cats, cat_to_name_path):
    print('Predictions for {}:'.format(img_path))

    if cat_to_name_path == None:
        for prob, cat in zip(probs, cats):
            print('{:30} : {:.3f}'.format(cat, prob))
    else:
        with open(cat_to_name_path, 'r') as file:
            cat_to_name = json.load(file)
        for prob, cat in zip(probs, cats):
            print('{:30} : {:.3f}'.format(cat_to_name[cat], prob))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str,
                        help='Classifier model path')
    parser.add_argument('-img', type=str,
                        help='Image path')
    parser.add_argument('-name', type=str, default=None,
                        help='Cat-to-name file path')
    parser.add_argument('-mean', type=float,
                        default=[0.485, 0.456, 0.406],
                        help='Dataset mean')
    parser.add_argument('-std', type=float,
                        default=[0.229, 0.224, 0.225],
                        help='Dataset standard deviation')
    parser.add_argument('-topk', type=int,
                        default=5, help='# predictions to display')
    parser.add_argument('-device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device')
    args = parser.parse_args()

    classifier = load_classifier(args.model)
    image = process_img(args.img, args.mean, args.std)
    probs, cats = predict(image, classifier, args.device, args.topk)
    show_prediction(args.img, probs, cats, args.name)

if (__name__ == '__main__'):
    main()
