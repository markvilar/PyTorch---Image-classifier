import argparse
import os
import torch
from torchvision import datasets as ds
from torchvision import transforms as tf
from torch import nn, optim
from torch.utils.data import DataLoader
from classifier import img_classifier, fc_net, freeze_parameters, save_classifier


def folder_count(path):
    count = 0
    for file in os.listdir(path):
        child = os.path.join(path, file)
        if os.path.isdir(child):
            count += 1
    return count


def data_preprocess(train_dir, valid_dir, test_dir, mean, std, batch_size):
    img_transforms = {'train' : tf.Compose([tf.RandomRotation(30),
                                            tf.RandomResizedCrop(224),
                                            tf.RandomHorizontalFlip(),
                                            tf.ToTensor(),
                                            tf.Normalize(mean, std)]),
                      'valid' : tf.Compose([tf.Resize(256),
                                            tf.CenterCrop(224),
                                            tf.ToTensor(),
                                            tf.Normalize(mean, std)]),
                      'test'  : tf.Compose([tf.Resize(256),
                                            tf.CenterCrop(224),
                                            tf.ToTensor(),
                                            tf.Normalize(mean, std)]) }

    img_datasets = {'train' : ds.ImageFolder(train_dir,
                                             transform=img_transforms['train']),
                    'valid' : ds.ImageFolder(valid_dir,
                                             transform=img_transforms['valid']),
                    'test' : ds.ImageFolder(test_dir,
                                            transform=img_transforms['test']) }

    dataloaders = {'train' : DataLoader(img_datasets['train'],
                                        batch_size=batch_size,
                                        shuffle=True),
                   'valid' : DataLoader(img_datasets['valid'],
                                        batch_size=batch_size,
                                        shuffle=True),
                   'test' : DataLoader(img_datasets['test'],
                                       batch_size=batch_size,
                                       shuffle=True) }

    return img_datasets, dataloaders


def train_classifer(classifier, dataloaders, device, epochs, print_every=20):
    '''
    Trains and validates the classifier
    Arguments
    ---------
    classifier:         class
    dataloaders:        torch.utils.data.DataLoader
    device:             str
    epochs:             int
    print_every:        int
    '''
    print("Training...")
    train_loss = 0
    steps = 0
    classifier.model.to(device)

    for epoch in range(epochs):
        classifier.model.train()
        for images, labels in dataloaders['train']:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            classifier.optimizer.zero_grad()

            # Loss and gradient calculation
            output = classifier.model(images)
            loss = classifier.criterion(output, labels)
            loss.backward()
            classifier.optimizer.step()
            train_loss += loss.item() / print_every

            # Validation
            if steps % print_every == 0:
                classifier.model.eval()
                valid_loss, valid_accuracy = validate_classifier(
                                                classifier, dataloaders, device)
                print("Epoch: {}/{}...".format(epoch+1, epochs),
                      "Training loss: {:.3f}...".format(train_loss),
                      "Validation loss: {:.3f}...".format(valid_loss),
                      "Validation accuracy: {:.3f}...".format(valid_accuracy),)
                train_loss = 0
                classifier.model.train()
    print("Training complete!")


def validate_classifier(classifier, dataloaders, device):
    '''
    Calculates the validation loss and accuracy for the classifier
    Arguments
    ---------
    classifier:         nn.Module
    dataloaders:        torch.utils.data.DataLoader
    device:             str
    '''
    steps = 0
    loss = 0
    accuracy = 0

    with torch.no_grad():
        for images, labels in dataloaders['valid']:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            output = classifier.model(images)
            loss += classifier.criterion(output, labels).item()
            pred_classes = torch.exp(output).max(dim=1)[1]
            corr_classes = (labels.data == pred_classes).type(torch.FloatTensor)
            accuracy += corr_classes.mean()

    return loss/steps, accuracy/steps


def test_classifier(classifier, dataloaders, device):
    '''
    Tests the classifier
    Arguments
    ---------
    classifier:         nn.Module
    dataloaders:        torch.utils.data.DataLoader
    device:             str
    '''
    print("Testing...")
    accuracy = 0
    steps = 0
    classifier.model.eval()
    classifier.model.to(device)

    for images, labels in dataloaders['test']:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        pred_classes = torch.exp(classifier.model(images)).max(dim=1)[1]
        corr_classes = (labels.data == pred_classes).type(torch.FloatTensor)
        accuracy += corr_classes.mean()

    print("Test accuracy: {:.3f}".format(accuracy/steps))
    return accuracy/steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-feature_net', type=str, default='vgg',
                        choices=['vgg', 'resnet', 'densenet'],
                        help='The feature detector network (CNN)')
    parser.add_argument('-hidden_sizes', type=int, nargs='+',
                        default=[512, 256, 128],
                        help='Number of neurons in the hidden layers')
    parser.add_argument('-learn_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('-dropout_prob', type=float, default=0.2,
                        help='Dropout probability')
    parser.add_argument('-data_dir', type=str,
                        help='Data directory')
    parser.add_argument('-mean', type=float, nargs='+',
                        default=[0.485, 0.456, 0.406],
                        help='Mean of the dataset')
    parser.add_argument('-std', type=float, nargs='+',
                        default=[0.229, 0.224, 0.225],
                        help='Standard deviation of the dataset')
    parser.add_argument('-batch_size', type=int, default=32,
                        help='The batch size of the training data')
    parser.add_argument('-epochs', type=int, default=7,
                        help='Number of training epochs')
    parser.add_argument('-device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='The device on which to do the training on')
    parser.add_argument('-save_path', type=str, default='default',
                        help='The file path where the classifier is saved')
    args = parser.parse_args()

    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    output_size = folder_count(train_dir)

    datasets, dataloaders = data_preprocess(train_dir, valid_dir, test_dir,
                                            args.mean, args.std,
                                            args.batch_size)
    classifier = img_classifier(args.hidden_sizes, output_size,
                                datasets['train'].class_to_idx,
                                args.feature_net, args.dropout_prob,
                                args.learn_rate)
    train_classifer(classifier, dataloaders, args.device, args.epochs)
    test_classifier(classifier, dataloaders, args.device)
    save_classifier(classifier, args.save_path)

if (__name__ == "__main__"):
    main();
