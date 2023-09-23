import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


#---------------------------------------------
#------------------ MNIST --------------------
#---------------------------------------------
from torchvision.datasets import MNIST

def load_mnist():
    # Define the data transformations
    data_transforms = {
        # use transforms.Compose to perform multiples transfomation at once
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
    }

    # Download and load the MNIST dataset
    train_dataset = MNIST(root='./data', train=True, transform=data_transforms['train'],download=True)
    test_dataset = MNIST(root='./data',  train=False, transform=data_transforms['val'],download=True)
    image_datasets ={
        'train':train_dataset,
        'val':test_dataset
    }
    # Create data loaders
    batch_size = 64
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                        for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return [dataloaders,dataset_sizes,class_names,device]


def show_images_mnist(dataloaders,class_names,device,num_img):
    fig = plt.figure()
    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        for j in range(num_img):
            ax = plt.subplot()
            ax.axis('off')
            ax.set_title(class_names[labels[j]])
            inp = inputs.data[j].numpy().transpose((1, 2, 0))
            mean = np.array(0.1307)
            std = np.array(0.3081)
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt.imshow(inp)
            plt.pause(0.001)
        break


#---------------------------------------------
#----------------- Cifar-10 ------------------
#---------------------------------------------
from torchvision.datasets import CIFAR10

def load_cifar_10():
    # Define the data transformations
    data_transforms = {
        # use transforms.Compose to perform multiples transfomation at once
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ]),
    }

    # Download and load the MNIST dataset
    train_dataset = CIFAR10(root='./data', train=True, transform=data_transforms['train'],download=True)
    test_dataset = CIFAR10(root='./data',  train=False, transform=data_transforms['val'],download=True)
    image_datasets ={
        'train':train_dataset,
        'val':test_dataset
    }
    # Create data loaders
    batch_size = 64
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                        for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return [dataloaders,dataset_sizes,class_names,device]

def show_images_cifar_10(dataloaders,class_names,device,num_img):
    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        #fig, axes = plt.subplots(1, 5, figsize=(32, 32))
        for j in range(num_img):
            fig = plt.figure(figsize=(1, 1))
            ax = plt.subplot()
            ax.axis('off')
            ax.set_title(class_names[labels[j]])
            inp = inputs.data[j].numpy().transpose((1, 2, 0))
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array( [0.2023, 0.1994, 0.2010])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt.imshow(inp)
            plt.pause(0.001)
        break




#---------------------------------------------
#----------------- Cifar-100 -----------------
#---------------------------------------------
from torchvision.datasets import CIFAR100

def load_cifar_100():
    # Define the data transformations
    data_transforms = {
        # use transforms.Compose to perform multiples transfomation at once
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ]),
    }

    # Download and load the MNIST dataset
    train_dataset = CIFAR100(root='./data', train=True, transform=data_transforms['train'],download=True)
    test_dataset = CIFAR100(root='./data',  train=False, transform=data_transforms['val'],download=True)
    image_datasets ={
        'train':train_dataset,
        'val':test_dataset
    }
    # Create data loaders
    batch_size = 64
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                        for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return [dataloaders,dataset_sizes,class_names,device]

def show_images_cifar_100(dataloaders,class_names,device,num_img):
    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        #fig, axes = plt.subplots(1, 5, figsize=(32, 32))
        for j in range(num_img):
            fig = plt.figure(figsize=(1, 1))
            ax = plt.subplot()
            ax.axis('off')
            ax.set_title(class_names[labels[j]])
            inp = inputs.data[j].numpy().transpose((1, 2, 0))
            mean = np.array([0.5071, 0.4867, 0.4408])
            std = np.array([0.2675, 0.2565, 0.2761])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt.imshow(inp)
            plt.pause(0.001)
        break
