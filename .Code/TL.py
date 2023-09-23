from torchvision import  transforms, datasets
import torch
import time
import copy
import os
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler


class TL_CNN:
    def __init__(self,model_CNN, epoch_train, epoch_fine_tune, data_dir=None, save_to_file = None):
        self.data_dir = data_dir
        self.save_to_file = save_to_file
        self.model_conv = model_CNN
        self.epoch_train = epoch_train
        self.epoch_fine_tune = epoch_fine_tune
        # Load dataset
        if(data_dir != None):
            self.dataloaders,self.dataset_sizes,self.class_names,self.device = self.load_data()
    #------------------------------------------------
    # Data loader functions 
    #------------------------------------------------

    #  Dataloader for classification profblem
    def load_data(self):
        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = { 
            # use transforms.Compose to perform multiples transfomation at once
            'train': transforms.Compose([
                # Crop a random portion of image and resize it to a given size.
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                transforms.ToTensor(),
                # Normalize a tensor image with mean and standard deviation.
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }


        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                    data_transforms[x])
                            for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                    shuffle=True, num_workers=4)
                        for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return [dataloaders,dataset_sizes,class_names,device]
    
    #------------------------------------------------
    #  General function to train a model
    #------------------------------------------------
    def train_model(self,model, criterion, optimizer, scheduler , num_epochs=7):

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        statics = []
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                statics.append(
                    {'epoch': epoch +1, 'phase': phase, 'acc': epoch_acc, 'loss': epoch_loss}
                )
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, statics
    

    #------------------------------------------------
    #  Replace Fully connected layer in model
    #------------------------------------------------
    def modif_fc(self,model, output):
      children_module = list(model.named_modules())
      last_module = children_module[len(children_module)-1][0]
      last_layers_name =[ '.'+ module  for module in last_module.split('.')]
      last_layers_name = last_layers_name[::-1]
      prev_module = None
      prev_module_name = ''
      i = 0
      module_name = last_module
      for module_last_layer_name in last_layers_name:
        module = model.get_submodule(module_name)
        module_name =  module_name.replace(module_last_layer_name, "")
        if(prev_module == None):
          module = nn.Linear(module.in_features, output)
        else:
          setattr(module,prev_module_name,prev_module)
        prev_module = module
        prev_module_name = module_last_layer_name.replace('.','')
      setattr(model,prev_module_name,prev_module)

    
    #------------------------------------------------
    #  General function to Apply TL on a CNN model 
    #------------------------------------------------
    def TL_classification_CNN(self):
        
        for param in self.model_conv.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        self.modif_fc( self.model_conv, len(self.class_names))
        self.model_conv = self.model_conv.to(self.device)

        criterion = nn.CrossEntropyLoss()

        # Observe that only parameters of final layer are being optimized as
        # opposed to before.
        param_optim =[ param for param in self.model_conv.parameters() if(param.requires_grad != False)]
        optimizer_conv = optim.SGD(param_optim, lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 5 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)
        print("-------------------------------------------- Training --------------------------------------------")
        # Train the model
        self.model_conv, self.statics_train = self.train_model(self.model_conv, criterion, optimizer_conv,
                                exp_lr_scheduler, num_epochs=self.epoch_train )
        if(self.save_to_file !=  None):
            torch.save(self.model_conv, self.save_to_file+' non fine_tuned')
        # unfreeze the feature layer
        print("-------------------------------------------- Fine-Tuning --------------------------------------------")
        for param in self.model_conv.parameters():
            param.requires_grad = True
        # Retrain the model (Fine-Tuning)
        self.model_conv, self.statics_fine_tune = self.train_model(self.model_conv, criterion, optimizer_conv,
                                exp_lr_scheduler, num_epochs=self.epoch_fine_tune)
        if(self.save_to_file !=  None):
            torch.save(self.model_conv, self.save_to_file)
        return self.model_conv