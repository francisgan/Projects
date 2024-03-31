import os
import numpy as np
import torch
import random
import re
import matplotlib.pyplot as plt


def check_directories(args):
    task_path = os.path.join(args.output_dir)
    if not os.path.exists(task_path):
        os.mkdir(task_path)
        print(f"Created {task_path} directory")
    
    folder = args.task
    
    save_path = os.path.join(task_path, folder)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(f"Created {save_path} directory")
    args.save_dir = save_path

    cache_path = os.path.join(args.input_dir, 'cache')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        print(f"Created {cache_path} directory")

    if args.debug:
        args.log_interval = max(args.n_epochs // 10, 1)

    return args

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def setup_gpus(args):
    n_gpu = 0  # set the default to 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    args.n_gpu = n_gpu
    if n_gpu > 0:   # this is not an 'else' statement and cannot be combined
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return args

def layer_wise_learning_rate(args,model):
    layers = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        layers.append(name)
    layers.reverse()

    lr = 1e-4  # Initial learning rate
    lr_mult = args.lr_mult  # Learning rate multiplier

    parameters = []
    # Initialize to identify the first component type
    prev_component_type = None

    for idx, name in enumerate(layers):
        # Determine the current component type based on the parameter name
        if name.startswith('encoder.layer'):
            component_type = 'encoder.layer.' + name.split('.')[2]  # Unique encoder layer ID
        elif name.startswith('pooler'):
            component_type = 'pooler'
        elif name.startswith('embeddings'):
            component_type = 'embeddings'
        else:
            component_type = 'other'

        # Apply the learning rate multiplier when transitioning between components
        if component_type != prev_component_type and prev_component_type is not None:
            lr *= lr_mult

        prev_component_type = component_type

        # Debug print statement; remove or comment out in production
        print(f'{idx}: lr = {lr:.6f}, {name}')

        # Add parameters with adjusted learning rate
        parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                        'lr': lr}]

    return parameters

def plot(train_acc,val_acc,dir_name = 'plots'):
    plt.figure(figsize=(12, 6))
    plt.plot(train_acc, label='Training accuracy', marker='o')
    plt.plot(val_acc, label='Validation accuracy', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.savefig(f'{dir_name}/training_validation_plots.png')

