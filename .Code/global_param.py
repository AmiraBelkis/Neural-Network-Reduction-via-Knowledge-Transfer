""" import data """
from Dataset_loader import load_cifar_100
dataloaders,dataset_sizes,class_names,device = load_cifar_100()


""" Pruning """
pruning_rate = 0.85
pruning_space = 0.05

"""-------------------------------"""


""" KD """
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler

kd_epoch = 5
lr = 0.01
stp = 7
prcntg_scheduler = 0.1
soft_citerion = nn.SmoothL1Loss()
hard_citerion = nn.CrossEntropyLoss()
soft_purcntg = 0.85 
hard_purcntg = 0.15

def Optimizer(indiv):
    return optim.SGD(indiv.parameters(), lr, momentum=0.9)
def Scheduler(optimizer_conv):
    return lr_scheduler.StepLR(optimizer_conv, step_size=stp, gamma=prcntg_scheduler)

""" fitness """
alpha = 0.45 
beta = 0.65
gama = 0.15

""" mutation """
proba_mutation = 0.1

""" performance condition """
acc_min = 0.9
sparsity_min = 0.85