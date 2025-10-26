# -*- coding: utf-8 -*-

# Import necessary libraries for PyTorch, data handling, optimization, and Weights & Biases (wandb) for experiment tracking
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
import random
import numpy as np

# Set seeds for reproducibility across runs
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

# Log in to Weights & Biases for tracking experiments
wandb.login()

# Import libraries for image augmentation using PIL
from PIL import Image, ImageEnhance, ImageOps

# Define Cutout augmentation class
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

# Define SubPolicy class for AutoAugment policies
class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        self.p1 = p1
        self.op1=operation1
        self.magnitude_idx1=magnitude_idx1
        self.p2 = p2
        self.op2=operation2
        self.magnitude_idx2=magnitude_idx2
        self.fillcolor=fillcolor
        self.init = 0

    def gen(self, operation1, magnitude_idx1, operation2, magnitude_idx2, fillcolor):
        # Define ranges for different augmentation magnitudes
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # Helper function for rotation with fill
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        # Dictionary of augmentation functions
        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude *
                                         random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude *
                                         random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude *
                                         img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude *
                                         img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if self.init == 0:
            self.gen(self.op1, self.magnitude_idx1, self.op2, self.magnitude_idx2, self.fillcolor)
            self.init = 1
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img

# Define ImageNetPolicy for AutoAugment on ImageNet
class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.
        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        # List of predefined sub-policies
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor)
        ]

    def __call__(self, img):
        # Randomly select a policy and apply it
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"

# Define CIFAR10Policy for AutoAugment on CIFAR10
class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        # List of predefined sub-policies for CIFAR10
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

# Function to load CIFAR10 dataset with data loaders
def GetCifar10(batchsize):
    """
    Load CIFAR10 dataset with training and test transformations.
    Args:
        batchsize (int): Batch size for data loaders.
    Returns:
        train_dataloader, test_dataloader
    """
    # Training transformations including augmentations
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  # CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16)
                                  ])
    # Test transformations
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # Load datasets
    train_data = datasets.CIFAR10('./data', train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10('./data', train=False, transform=trans, download=True)

    # Create data loaders
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader

# Define VGG model class
class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # fixed output size
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# Activation function can be passed as a parameter, default it is relu.
def make_layers(cfg, batch_norm=False, activation='relu'):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # Select activation based on input parameter
            if activation == 'relu':
              act = nn.ReLU(inplace=True)
            elif activation == 'sigmoid':
              act = nn.Sigmoid()
            elif activation == 'tanh':
              act = nn.Tanh()
            elif activation == 'silu':
              act = nn.SiLU()
            elif activation == 'gelu':
              act = nn.GELU()
            else:
              raise ValueError(f"Unsupported activation function: {activation}")

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), act]
            else:
                layers += [conv2d, act]
            in_channels = v
    return nn.Sequential(*layers)

# Function to create VGG model
def vgg(cfg, num_classes=10, batch_norm=True, activation='relu'):
    """
    Create VGG model with given configuration.
    Args:
        cfg (list): Layer configuration.
        num_classes (int): Number of output classes.
        batch_norm (bool): Use batch normalization.
        activation (str): Activation function.
    Returns:
        VGG: Instantiated VGG model.
    """
    return VGG(make_layers(cfg, batch_norm=batch_norm, activation=activation), num_classes=num_classes)

# VGG-6 configuration
cfg_vgg6 = [64, 64, 'M', 128, 128, 'M']

# Example model instantiation and print (for debugging)
model = vgg(cfg_vgg6, num_classes=10, batch_norm=True, activation='relu')
print(model)

# Evaluation function
def eval(model,data_loader, criterion):
    """
    Evaluate the model on a given data loader.
    Args:
        model: The model to evaluate.
        data_loader: DataLoader for evaluation.
        criterion: Loss function.
    Returns:
        acc (float): Accuracy.
        avg_loss (float): Average loss.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    acc = 100. * correct / total
    avg_loss = total_loss / len(data_loader)
    return acc, avg_loss #Return both accuracy and loss

# Example data loaders (for debugging)
train_loader,test_loader = GetCifar10(64)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example model, criterion, optimizer (for debugging)
model = vgg(cfg_vgg6, num_classes=10, batch_norm=True, activation='relu').to(device)

criterion = nn.CrossEntropyLoss()
lr = 0.001
epochs = 100
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training function with model saving
def train_model(model,epochs,optimizer,train_loader,test_loader, config):
    """
    Train the model and log metrics to wandb. Save the best model based on test accuracy.
    Args:
        model: The model to train.
        epochs: Number of epochs.
        optimizer: Optimizer.
        train_loader: Training DataLoader.
        test_loader: Test DataLoader.
        config: Configuration dictionary.
    """
    best_test_acc = 0.0
    model.train()
    for epoch in range(config.epochs):
      running_loss = 0.0
      for data, target in train_loader:
          data, target = data.to(device), target.to(device)
          optimizer.zero_grad()
          outputs = model(data)
          loss = criterion(outputs, target)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

      # Evaluate on train and test sets
      train_acc, train_loss_eval = eval(model,train_loader,criterion)
      test_acc, test_loss = eval(model,test_loader,criterion)

      # Log metrics to wandb
      wandb.log({"epoch": epoch, "train_loss": running_loss/len(train_loader), "train_acc": train_acc, "test_acc": test_acc, "test_loss": test_loss})
      print(f"Epoch {epoch} - Train_Loss: {running_loss/len(train_loader):.4f} , Train_acc: {train_acc}, Test_acc : {test_acc}")

      # Save best model if improved
      if test_acc > best_test_acc:
          best_test_acc = test_acc
          print(f"New best model found! Accuracy: {best_test_acc:.2f}%. Saving model...")
          # Define the save path within the W&B run directory
          model_save_path = f"{wandb.run.dir}/best_model.pth"
          # Save the model's parameters
          torch.save(model.state_dict(), model_save_path)

    print("--- Training finished for this configuration ---")

# Function to get optimizer based on name
def get_optimizer(model, optimizer_name, lr):
    """
    Return optimizer based on name.
    Args:
        model: Model whose parameters to optimize.
        optimizer_name (str): Name of optimizer.
        lr (float): Learning rate.
    Returns:
        Optimizer instance.
    """
    if optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'nesterov-sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'nadam':
        return optim.NAdam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

# Sweep training function
def train_sweep():
    """
    Function to run a single sweep configuration, initialize model, optimizer, and train.
    """
    wandb.init(project="2-vgg6-cifar10", config={}, entity="abhishek-chaudhary91-iit-madras")
    config = wandb.config
    train_loader, test_loader = GetCifar10(config.batch_size)
    model = vgg(cfg_vgg6, num_classes=10, batch_norm=True, activation=config.activation).to(device)
    optimizer = get_optimizer(model, config.optimizer, config.lr)
    train_model(model, config.epochs, optimizer, train_loader, test_loader, config)

# Sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'test_acc', 'goal': 'maximize'},
    'parameters': {
        'activation': {'values': ['relu', 'sigmoid', 'tanh', 'silu', 'gelu']},
        'optimizer': {'values': ['sgd', 'nesterov-sgd', 'adam', 'adagrad', 'rmsprop', 'nadam']},
        'lr': {'distribution': 'log_uniform_values', 'min': 0.0001, 'max': 0.01},
        'batch_size': {'distribution': 'q_log_uniform_values', 'q': 8, 'min': 32, 'max': 128},
        'epochs': {'values': [50, 100]}
    }
}

# Create sweep and run agent
sweep_id = wandb.sweep(sweep_config, project="2-vgg6-cifar10", entity="abhishek-chaudhary91-iit-madras")

wandb.agent(sweep_id, function=train_sweep, count=20)  # Limit to 20 runs for now
