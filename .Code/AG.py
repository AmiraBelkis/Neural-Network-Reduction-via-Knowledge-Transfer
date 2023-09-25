
#from my_lib import *
import random
import string
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import matplotlib.colors as colors
import numpy as np
from copy import deepcopy
from global_param import *


#  Generate a random name for saving child models in the drive 
def generate_random_name(length=8):
    letters = string.ascii_lowercase
    random_name = ''.join(random.choice(letters) for _ in range(length))
    return random_name
""" Reconstruct the input image from a 3 channel input data"""
def show_input_img(image):
  mean = np.array([0.485, 0.456, 0.406] )
  std = np.array([0.229, 0.224, 0.225])
  normalized_image = image  # Replace with your tensor
  original_image = normalized_image * std[:, None, None] + mean[:, None, None]
  original_image = np.transpose(original_image, (1, 2, 0))
  plt.imshow(original_image)
  plt.axis('off')
  plt.show()

""" View the parameters (kernels or masks) of a convolutional layer as images
'Each channel has its own image'
Where the 0 value is represented by white pixels, positive values by bright red,
and negative values by dark blue
And save those images to the 'save_to' directory
Ex: visualize_filter(conv_layer.weight ,'/filtres/weights')
"""
def visualize_filter(conv_layer_weight ,save_to = None):
    filters = conv_layer_weight.data
    num_filters = filters.size(0)
    num_channels = filters.size(1)

    # Determine the number of rows and columns for the grid
    rows = int(num_filters ** 0.5)
    cols = (num_filters // rows) + int(num_filters % rows > 0)
    for j in range(num_channels):
      print('-------------------------- channel {} --------------------------'.format(j))
      # Plot each filter's channels as separate images in the grid
      fig, axes = plt.subplots(rows, cols)
      colors_list = ['#023047','#FFFFFF','#d90429'] #['#FF2171','#FF90BB','#FFFAD7'] #['#130637', '#2124b5', '#91cafd']
      #cmap = colors.ListedColormap(colors_list)
      cmap = colors.LinearSegmentedColormap.from_list('custom_cmap', colors_list)
      # Set the boundaries for the color map
      bounds = [-10, 0 , 0.0001, 10]
      norm = colors.BoundaryNorm(bounds, cmap.N)
      # Normalize the data values
      data_np = filters.numpy()
      vmin = np.min(data_np)
      vmax = np.max(data_np)
      if (np.abs(vmin) > np.abs(vmax)) and (vmin < 0):
        vmax = -1 * vmin
      elif (np.abs(vmin) < np.abs(vmax)) and (vmax > 0):
        vmin = -1 * vmax
      elif ( vmax == 0 ):
        vmax = np.abs(vmin)
      elif (vmin == 0):
        vmin = -1 * np.abs(vmax)
      norm = colors.Normalize(vmin=vmin, vmax=vmax)
      for i, ax in enumerate(axes.flatten()):
          if i < num_filters:
              filter_i = filters[i]
              channel_j = filter_i[j]
              im = ax.imshow(channel_j.detach().numpy(), cmap=cmap, norm=norm)
              ax.axis('off')
          else:
              ax.axis('off')
      if (save_to != None):
        plt.savefig(save_to+'channel_'+str(j)+'.png')
      #plt.colorbar(orientation='horizontal', label='Pixel Value')
      # Add a colorbar for the entire figure
      fig.colorbar(axes[0, 0].images[0], ax=axes, orientation='horizontal', label='Pixel Value')

      plt.show()

"""# Visualise training statistic"""
def show_statics_training(statics, file_name = None):
  # Separate the data into accuracy and loss
  acc_data_train = [entry['acc'] for entry in statics if entry['phase'] == 'train']
  acc_data_val = [entry['acc'] for entry in statics if entry['phase'] == 'val']
  loss_data_train = [entry['loss'] for entry in statics if entry['phase'] == 'train']
  loss_data_val = [entry['loss'] for entry in statics if entry['phase'] == 'val']

  cuda = False
  try:
    cuda = (acc_data_train[0].device.type == 'cuda')
  except:
    cuda = False
  if(cuda):
    acc_data_train = [ tensor.cpu() for tensor in acc_data_train ]

    acc_data_val = [ tensor.cpu() for tensor in acc_data_val ]


  # Create the figure and axes
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

  # Plot accuracy
  ax1.plot(acc_data_train, label='Train Accuracy')
  ax1.plot(acc_data_val, label='Validation Accuracy')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Accuracy')
  ax1.legend()

  # Plot loss
  ax2.plot(loss_data_train, label='Train Loss')
  ax2.plot(loss_data_val, label='Validation Loss')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Loss')
  ax2.legend()

  # Adjust layout and save the figure
  plt.tight_layout()
  if (file_name != None):
    plt.savefig(file_name+'_statics_graph.png')
  plt.show()
"""# Visualise training statistic"""

""" ##Gennerate initial population  """
"""
this method generates the initial population by performing random unstructured pruning on the base model
"""
# save_to must ends with '/'
def generat_population(model_base, N,pruning_rate,pruning_space, save_to):
  population = []
  for i in range(N): # we will generate N individus
    model_copy = deepcopy(model_base)
    for name, layer in model_copy.named_modules():
        # the pruning is performed in a layer-wise way 
        pruning_rate_layer = random.uniform(pruning_rate - pruning_space, pruning_rate + pruning_space)
        if (pruning_rate_layer <= 0) or (pruning_rate_layer >= 1):
          pruning_rate_layer = 0.5
        # each layer has its own pruning rate that is generated randomly between [pruning_rate - pruning_space, pruning_rate + pruning_space] 
        if( hasattr(layer, 'weight') )  :
          prune.random_unstructured(layer, name='weight', amount=pruning_rate_layer)
    print('     - individual {} ----------- 100%'.format(i))
    torch.save(model_copy, save_to + 'indiv'+str(i))
    population.append(model_copy)
  return population


""" ## Perform KD """
""" 
this method we will train the individual using KD concept 
ie the loss will be calculated according to the label in the training data 
and the output of the base model.
"""
def train_model_kd_mix(model,model_teacher, criterion, optimizer, scheduler , num_epochs=7):
    since = time.time()

    model_teacher.eval()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
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
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass with the teacher model
                with torch.no_grad():
                    soft_targets = model_teacher(inputs)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, soft_targets, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            statics.append(
                {'epoch': epoch +1, 'phase': phase, 'acc': epoch_acc, 'loss': epoch_loss}
            )

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    setattr(model,'best_acc',best_acc)
    setattr(model,'best_loss',best_loss)
    return model, statics

"""--------------"""

""" define a new loss function class to calculate the loss using 2 diifrent loss functions:
the 1st function (soft_target_loss) will calculate the loss between the student and the teacher model 
the 2nd function (hard_target_loss) will calculate the loss between the student and the labels from the training dataset  
"""
class SoftHardTargetLoss(nn.Module):
    def __init__(self,soft_target_loss, hard_target_loss,alpha,beta):
      super(SoftHardTargetLoss, self).__init__()
      self.soft_target_loss = soft_target_loss
      self.hard_target_loss = hard_target_loss
      self.alpha = alpha
      self.beta = beta

    def forward(self, student_outputs, teacher_outputs, labels):
      loss_soft = self.soft_target_loss(student_outputs,teacher_outputs)
      loss_hard = self.hard_target_loss(student_outputs,labels)
      loss = self.alpha * loss_soft + self.beta * loss_hard
      return loss


""" ## calculate fitness """

""" 
this method evaluate the performance of a model,
it returns the accurancy and the loss  
"""
def evaluation_2(model, criterion):
  # Set the model to evaluation mode
  model = model.to(device)
  model.eval()
  total_loss = 0.0
  total_correct = 0
  total_samples = 0

  # Iterate over the test/validation dataset
  for inputs, labels in dataloaders['val']:
      # Forward pass
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)

      # Calculate the loss
      loss = criterion(outputs, labels)
      total_loss += loss.item()

      # Get the predicted labels
      _, predicted = torch.max(outputs, dim=1)

      # Update the counts
      total_correct += (predicted == labels).sum().item()
      total_samples += labels.size(0)

  # Compute the accuracy
  accuracy = total_correct / total_samples

  # Compute the average loss
  average_loss = total_loss / len(dataloaders['val'])

  print("Accuracy: {:.2f}%".format(accuracy * 100))
  print("Loss: {:.4f}".format(average_loss))

  return accuracy, average_loss
"""--------------"""


""" 
this method calculate the average % of the 0 weights in a model  
"""
def calculate_sparsity(model):
  average_sparsity = 0
  nb_lyr = 0
  for name, layer in model.named_modules():
    if(hasattr(layer, 'weight')):
      nb_lyr += 1
      sparsity = 1.0 - torch.sum(layer.weight != 0) / float(layer.weight.nelement())
      average_sparsity += sparsity

  average_sparsity = average_sparsity / nb_lyr
  #print('Average sparsity: {}'.format(average_sparsity))
  return average_sparsity
"""-----------------"""

""" fitness is the metric of individual and it is calculated from the accurancy and loss of the model
plus the % of sparsity (the 0 values in a model's weights) """
def calculate_fitness(model,accuracy, average_loss, alpha  , beta , gama ):
  #accuracy, average_loss = evaluation_2(model, criterion)
  average_sparsity = calculate_sparsity(model)
  fitness = (alpha * accuracy) + (beta * average_sparsity) + (gama / average_loss)
  return fitness
"""-----------------"""


""" ## Select parents """

""" the individual with the hieghest fitness will be selected for the Croisement """
def sort_population (population, alpha, beta , gama ) :
  population.sort(key=lambda indiv : calculate_fitness(indiv,indiv.best_acc,indiv.best_loss,alpha,beta,gama), reverse=True)
  return None

def select_parents_elits(population,alpha, beta, gama) :
  sort_population(population,alpha,beta,gama)
  elits = []
  NC = []
  indice_fin = int (len(population)*0.5)
  elits = elits + (population[: indice_fin])
  NC = NC + (population[indice_fin :])

  # swap some individus to keep diversity 
  rand = random.uniform(0.1, 0.25) # swap between 10% and 25% of the population
  nb_individ_swap = int(indice_fin * rand)
  if(nb_individ_swap == 0):
    nb_individ_swap = 1 # we swap at least one individu
  for _ in range(nb_individ_swap):
    indx = random.randint(1, indice_fin - 1)
    indiv_ = NC[indx]
    NC[indx] = elits[indx]
    elits[indx] = indiv_ 

  return elits, NC
"""-----------------"""


""" ## Crossing """
""" swaping 2 layer => weights and mask of layer 1 will be the weights and mask of layer 2 """
def swap_param(layer_1,layer_2):

   # ---------- weight ----------
  if(hasattr(layer_2, 'weight') and layer_2.weight != None):
    cloned_weight = layer_2.weight.data.clone()
    if(hasattr(layer_1, 'weight') and layer_1.weight != None):
      layer_2.weight.data = layer_1.weight.data
      layer_1.weight.data = cloned_weight

  # ---------- weight_mask ----------
  if(hasattr(layer_2, 'weight_mask') and layer_2.weight_mask != None):
    cloned_weight_mask = layer_2.weight_mask.data.clone()
    if(hasattr(layer_1, 'weight_mask') and layer_1.weight_mask != None):
      layer_2.weight_mask.data = layer_1.weight_mask.data
      layer_1.weight_mask.data = cloned_weight_mask

  # ---------- weight_orig ----------
  if(hasattr(layer_2, 'weight_orig') and layer_2.weight_orig != None):
    cloned_weight_orig = layer_2.weight_orig.data.clone()
    if(hasattr(layer_1, 'weight_orig') and layer_1.weight_orig != None):
      layer_2.weight_orig.data = layer_1.weight_orig.data
      layer_1.weight_orig.data = cloned_weight_orig

  # ---------- Bias ----------
  if(hasattr(layer_2, 'bias') and layer_2.bias != None):
    cloned_bias = layer_2.bias.data.clone()
    if(hasattr(layer_1, 'bias') and layer_1.bias != None):
      layer_2.bias.data = layer_1.bias.data
      layer_1.bias.data = cloned_bias

"""----------------------"""


""" Perform croisement between 2 individuals using mask
if mask[i] = 1 then swap the ieme layers 
"""
def crossing(indiv_1,indiv_2):
  modules_with_weights_1 = [module for _ , module in indiv_1.named_modules() if  (hasattr(module,'weight'))]
  modules_with_weights_2 = [module for _ , module in indiv_2.named_modules() if  (hasattr(module,'weight'))]
  mask = []
  for i in range(len(modules_with_weights_2)):
    m = np.random.choice([0,1])
    mask.append(m)
    if m == 1 :
      layer_1 = modules_with_weights_1[i]
      layer_2 = modules_with_weights_2[i]
      swap_param(layer_1,layer_2)
  # swap at least one layer 
  mask_ = np.array(mask)
  if((mask_==1).all() or (mask_==0).all()):
    indx = random.randint(0, len(mask)- 1)
    layer_1 = modules_with_weights_1[indx]
    layer_2 = modules_with_weights_2[indx]
    mask[indx] = 0 if (mask_==1).all() else 1
    swap_param(layer_1,layer_2)
  return indiv_1,indiv_2,mask
"""----------------------"""


""" ## Mutation """
def mutation(model):
  modules_with_weights = [module for _ , module in model.named_modules() if (hasattr(module,'weight')) and (hasattr(module,'weight_mask'))]
  layer_indx = random.randint(0, len(modules_with_weights)-1)
  layer = modules_with_weights[layer_indx]
  mask = layer.weight_mask.data
  j = 0
  indx = []
  for _ in mask.size():
    rand = random.randint(0, mask.size(j)-1)
    indx.append(rand)
    j += 1
  mask[tuple(indx)] = torch.tensor(0.0)
"""----------------------"""


""" ## General method to perform AG """
def AG_method(model_base, save_to, nb_itr = 2, N = 4 ,pruning_rate = 0.8, pruning_space = 0.15, kd_epoch = 5):
    model_base = model_base.to(device)
    # Generate initial population
    print('Generating the initial population :')
    population = generat_population(model_base, N,pruning_rate,pruning_space, save_to)
    # remove pruning to be able to use deepcopy protocol
    
    # use CrossEntropyLoss for labels loss and SmoothL1Loss for teacher ouput loss
    print('Performing KD on the initial population :')
    statics_population_kd = []
    i = 0
    for indiv in population:
        i= i+1
        indiv = indiv.to(device)

        # Calculate the loss
        criterion = SoftHardTargetLoss(soft_citerion,hard_citerion, soft_purcntg, hard_purcntg)

        optimizer_conv = Optimizer(indiv)

        exp_lr_scheduler = Scheduler(optimizer_conv)
        print("------------------------------ Training {} ---------------------------------".format(i))
        # Train the model
        indiv, statics = train_model_kd_mix(indiv,model_base, criterion, optimizer_conv,exp_lr_scheduler, num_epochs=kd_epoch)
        statics_population_kd.append(statics)
        torch.save(indiv, save_to+ 'kd/indiv0'+str(i))
    print("------------------------------------------------------------")
    print('Show training statistic:')
    print("________________________")

    i = 0
    for statics in statics_population_kd:
        i = i+1
        path = save_to + 'kd/training/0'
        show_statics_training(statics,path + str(i))

    itr = 0
    performance = False
    while ((itr < nb_itr) and (not performance) ):
        itr = itr + 1
        print('/*-----------------------------*/')
        print('/*         iteration {}        */'.format(itr))
        print('/*----------------------------*/')
        print('   * Select parent for crossing:')
        parents_, NC = select_parents_elits(population,alpha,beta,gama)
        print('       - Selection --------------- 100%')
        print('   * Offspring:')
        children = []
        Next_generation = []
        t = 0
        p = 0
        while (len(parents_) > 1):
          # Select a pair of parents
          selected_objects = random.sample(parents_, 2)
          parent_1,parent_2 = selected_objects
          parents_.remove(parent_1)
          parents_.remove(parent_2)
          # Save th parents to use in the recovery code
          p += 1
          torch.save(parent_1,save_to +'parents/(itr-'+str(itr)+')parent_'+str(p))
          p += 1
          torch.save(parent_2,save_to+'parents/(itr-'+str(itr)+')parent_'+str(p))
          
          # Select a pair of NCs to use as children
          selected_objects = random.sample(NC, 2)
          child_1,child_2 = selected_objects
          NC.remove(child_1)
          NC.remove(child_2)
          child_1.load_state_dict(deepcopy(parent_1.state_dict()))
          child_2.load_state_dict(deepcopy(parent_2.state_dict()))
          print('       - Crossover:')

          # perform crossing
          child_1,child_2,mask = crossing(child_1,child_2)
          print('           mask :',mask)

          # perform mutation
          proba = random.uniform(0.0, 1.0)
          if(proba>0.9):
            print('       - mutation:')
            mutation(child_1)
            mutation(child_2)

          children.append(child_1)
          children.append(child_2)
          t += 1
          torch.save(child_1,save_to+'children/(itr-'+str(itr)+')child_'+str(t))
          t += 1
          torch.save(child_2,save_to+'children/(itr-'+str(itr)+')child_'+str(t))

          # add parents and children to the next generation
          Next_generation.append(parent_1)
          Next_generation.append(parent_2)
          Next_generation.append(child_1)
          Next_generation.append(child_2)
        """----------------------------"""
      
        print('       - Performing KD on the children :')
        statics_population_kd = []
        i = 0
        for indiv in children:
            i= i+1
            indiv = indiv.to(device)

            # Calculate the loss
            criterion = SoftHardTargetLoss(soft_citerion,hard_citerion, soft_purcntg, hard_purcntg)

            optimizer_conv = Optimizer(indiv)

            exp_lr_scheduler = Scheduler(optimizer_conv)
            print("------------------------------ Training {} ---------------------------------".format(i))
            # Train the model
            indiv, statics = train_model_kd_mix(indiv,model_base, criterion, optimizer_conv,exp_lr_scheduler, num_epochs=kd_epoch)
            statics_population_kd.append(statics)
            torch.save(indiv, save_to+ 'kd/(itr-'+str(itr)+')indiv'+str(i))
        
        print('       - Show training statistic:')
        i = 0
        for statics in statics_population_kd:
            i = i+1
            path = save_to + 'kd/training/(itr-'+str(itr)+')'
            show_statics_training(statics,path + str(i))
        population = Next_generation
        sort_population(population,alpha,beta,gama)
        if(calculate_sparsity(population[0])>sparsity_min)and (population[0].best_acc>acc_min):
           performance = True
    print('Select the winning ticket from final population')
    sort_population(population,alpha,beta,gama)
    winning_ticket = population[0]
    torch.save(winning_ticket, save_to + 'winning_ticket')
    print('  - Selection --------------- 100%')
    return winning_ticket
"""----------------------"""


def AG_extra_itr(model_base,population, save_to, itr,nb_itr,kd_epoch):
  performance = False
  while ((itr < nb_itr) and (not performance) ):
      itr = itr + 1
      print('/*-----------------------------*/')
      print('/*         iteration {}        */'.format(itr))
      print('/*----------------------------*/')
      print('   * Select parent for crossing:')
      parents_, NC = select_parents_elits(population,alpha,beta,gama)
      print('       - Selection --------------- 100%')
      print('   * Offspring:')
      children = []
      Next_generation = []
      t = 0
      p = 0
      while (len(parents_) > 1):
        # Select a pair of parents
        selected_objects = random.sample(parents_, 2)
        parent_1,parent_2 = selected_objects
        parents_.remove(parent_1)
        parents_.remove(parent_2)
        # Save th parents to use in the recovery code
        p += 1
        torch.save(parent_1,save_to +'parents/(itr-'+str(itr)+')parent_'+str(p))
        p += 1
        torch.save(parent_2,save_to+'parents/(itr-'+str(itr)+')parent_'+str(p))
        
        # Select a pair of NCs to use as children
        selected_objects = random.sample(NC, 2)
        child_1,child_2 = selected_objects
        NC.remove(child_1)
        NC.remove(child_2)
        child_1.load_state_dict(deepcopy(parent_1.state_dict()))
        child_2.load_state_dict(deepcopy(parent_2.state_dict()))
        print('       - Crossover:')

        # perform crossing
        child_1,child_2,mask = crossing(child_1,child_2)
        print('           mask :',mask)

        # perform mutation
        proba = random.uniform(0.0, 1.0)
        if(proba>(1- proba_mutation)):
          print('       - mutation:')
          mutation(child_1)
          mutation(child_2)

        children.append(child_1)
        children.append(child_2)
        t += 1
        torch.save(child_1,save_to+'children/(itr-'+str(itr)+')child_'+str(t))
        t += 1
        torch.save(child_2,save_to+'children/(itr-'+str(itr)+')child_'+str(t))

        # add parents and children to the next generation
        Next_generation.append(parent_1)
        Next_generation.append(parent_2)
        Next_generation.append(child_1)
        Next_generation.append(child_2)
      """----------------------------"""
    
      print('       - Performing KD on the children :')
      statics_population_kd = []
      i = 0
      for indiv in children:
          i= i+1
          indiv = indiv.to(device)

          # Calculate the loss
          criterion = SoftHardTargetLoss(soft_citerion,hard_citerion, soft_purcntg, hard_purcntg)

          optimizer_conv = Optimizer(indiv)

          exp_lr_scheduler = Scheduler(optimizer_conv)
          print("------------------------------ Training {} ---------------------------------".format(i))
          # Train the model
          indiv, statics = train_model_kd_mix(indiv,model_base, criterion, optimizer_conv,exp_lr_scheduler, num_epochs=kd_epoch)
          statics_population_kd.append(statics)
          torch.save(indiv, save_to+ 'kd/(itr-'+str(itr)+')indiv'+str(i))
      
      print('       - Show training statistic:')
      i = 0
      for statics in statics_population_kd:
          i = i+1
          path = save_to + 'kd/training/(itr-'+str(itr)+')'
          show_statics_training(statics,path + str(i))
      population = Next_generation
      sort_population(population,alpha,beta,gama)
      if(calculate_sparsity(population[0])>sparsity_min)and (population[0].best_acc>acc_min):
          performance = True
  print('Select the winning ticket from final population')
  sort_population(population,alpha,beta,gama)
  winning_ticket = population[0]
  torch.save(winning_ticket, save_to + 'winning_ticket')
  print('  - Selection --------------- 100%')
  return winning_ticket
