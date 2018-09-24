import torch
from torch import nn, optim
from torchvision import models


class img_classifier(nn.Module):
    '''
    The CNN (feature module + classifier module) with the additional information
    needed to reconstruct the network

    Member variables
    ----------------
    model:              nn.Module
    optimizer:          nn.Module
    hidden_sizes:       list of ints
    output_size:        int
    cat_to_idx:       dict {str: int}
    feature_net:        str
    dropout_prob:       float
    learn_rate:         float
    criterion:          nn.Module
    '''
    def __init__(self, hidden_sizes, output_size, cat_to_idx, feature_net,
                 dropout_prob, learn_rate):
        super().__init__()

        # Downloads feature network, freezes parameters, attaches classifier
        if feature_net == 'densenet':
            self.model = models.densenet121(pretrained=True)
            self.model = freeze_parameters(self.model)
            input_size = self.model.classifier.in_features
            self.model.classifier = fc_net(input_size, hidden_sizes,
                                           output_size, dropout_prob)
            self.optimizer = optim.Adam(self.model.classifier.parameters(),
                                        lr=learn_rate)

        elif feature_net == 'resnet':
            self.model = models.resnet18(pretrained=True)
            self.model = freeze_parameters(self.model)
            input_size = self.model.fc.in_features
            self.model.fc = fc_net(input_size, hidden_sizes, output_size,
                                   dropout_prob)
            self.optimizer = optim.Adam(self.model.fc.parameters(),
                                        lr=learn_rate)

        elif feature_net == 'vgg':
            self.model = models.vgg16(pretrained=True)
            self.model = freeze_parameters(self.model)
            input_size = self.model.classifier[0].in_features
            self.model.classifier = fc_net(input_size, hidden_sizes,
                                           output_size, dropout_prob)
            self.optimizer = optim.Adam(self.model.classifier.parameters(),
                                        lr=learn_rate)

        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.cat_to_idx = cat_to_idx
        self.feature_net = feature_net
        self.dropout_prob = dropout_prob
        self.learn_rate = learn_rate
        self.criterion = nn.NLLLoss()


class fc_net(nn.Module):
    '''
    The fully connected module of the CNN
    Member variables
    ----------------
    hidden_layers:      nn.ModuleList
    output_layer:       nn.Linear
    dropout_prob:       float
    requires_grad:      bool
    '''
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob):
        super().__init__()

        # Defines layers
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])

        for size_1, size_2 in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self.hidden_layers.extend([nn.Linear(size_1, size_2)])

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.requires_grad = True

    # Feed-forward
    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = nn.functional.relu(hidden_layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return nn.functional.log_softmax(x, dim=1)


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def save_classifier(classifier, path):
    checkpoint = {'model_state_dict' : classifier.model.state_dict(),
                  'optimizer_state_dict' : classifier.optimizer.state_dict(),
                  'hidden_sizes' : classifier.hidden_sizes,
                  'output_size' : classifier.output_size,
                  'cat_to_idx' : classifier.cat_to_idx,
                  'feature_net' : classifier.feature_net,
                  'dropout_prob' : classifier.dropout_prob,
                  'learn_rate' : classifier.learn_rate}
    print("Saving classifier to {}".format(path))
    torch.save(checkpoint, path)
    print("Classifier saved successfully!")




def load_classifier(path):
    checkpoint = torch.load(path)
    print("Loading classifier from {} ...".format(path))
    classifier = img_classifier(checkpoint['hidden_sizes'],
                                    checkpoint['output_size'],
                                    checkpoint['cat_to_idx'],
                                    checkpoint['feature_net'],
                                    checkpoint['dropout_prob'],
                                    checkpoint['learn_rate'])
    classifier.model.load_state_dict(checkpoint['model_state_dict'])
    classifier.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Classifier loaded successfully!")
    return classifier
