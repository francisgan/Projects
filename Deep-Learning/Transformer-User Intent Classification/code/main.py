import os, sys, pdb
import numpy as np
import random
import torch

import math

from tqdm import tqdm as progress_bar

from utils import set_seed, setup_gpus, check_directories,plot
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import IntentModel, SupConModel, CustomModel, Classifier
from torch import nn

from loss import SupConLoss

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
from scipy.stats import mode
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from torch.optim.swa_utils import SWALR, AveragedModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def baseline_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    run_eval(args, model, datasets, tokenizer, split='validation',criterion=criterion)
    run_eval(args, model, datasets, tokenizer, split='test',criterion=criterion)
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'])
    # task2: setup model's optimizer_scheduler if you have
    
    all_valid_acc = []    
    all_train_acc = []
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in enumerate(progress_bar(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)

            logits = model(inputs,labels)
            loss = criterion(logits, labels)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            model.scheduler.step()  # Update learning rate schedule
            
            model.zero_grad()
            losses += loss.item()
        avg_loss = losses / len(train_dataloader)
        if True:
            with open(args.file_name, 'a') as log_file:
                log_file.write(f'\nepoch: {epoch_count}, losses: {avg_loss}\n')
                
        val_acc = run_eval(args, model, datasets, tokenizer, split='validation',criterion=criterion, epoch_count=epoch_count)
        train_acc = run_eval(args, model, datasets, tokenizer, split='train',criterion=criterion, epoch_count=epoch_count)
        
        all_valid_acc.append(val_acc)
        all_train_acc.append(train_acc)

        print('epoch', epoch_count, '| losses:', avg_loss)
        
    run_eval(args, model, datasets, tokenizer, split='test',criterion=criterion)
    plot(all_train_acc, all_valid_acc)
def custom_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    run_eval(args, model, datasets, tokenizer, split='validation',criterion=criterion)
    run_eval(args, model, datasets, tokenizer, split='test',criterion=criterion)
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'])
    # task2: setup model's optimizer_scheduler if you have

    if args.early_stop:
        patience_counter = 0
        best_acc = 0
        early_stopping_patience = 2
            

    optimizer = model.optimizer
    scheduler = model.scheduler
    all_valid_acc = []    
    all_train_acc = []
    swa_model = AveragedModel(model)
    swa_start = args.swa_start
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr, anneal_strategy='linear')
        

    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in enumerate(progress_bar(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)

            logits = model(inputs,labels)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()  # backprop to update the weights
            model.zero_grad()
            losses += loss.item()
            
#         scheduler.step()  

        if epoch_count >= swa_start:
            print("SWA step")
            swa_scheduler.step()
        else:
            scheduler.step()

        if epoch_count >= swa_start:
            swa_model.update_parameters(model)

            
        avg_loss = losses / len(train_dataloader)
        if epoch_count % args.log_interval == 0 or epoch_count == args.n_epochs - 1:
            with open(args.file_name, 'a') as log_file:
                log_file.write(f'\nepoch: {epoch_count}, losses: {avg_loss}\n')
        train_acc = run_eval(args, model, datasets, tokenizer, split='train',criterion=criterion, epoch_count=epoch_count)
        all_train_acc.append(train_acc)
        valid_acc = run_eval(args, model, datasets, tokenizer, split='validation',criterion=criterion, epoch_count=epoch_count)
        all_valid_acc.append(valid_acc)
        if valid_acc > best_acc:
            best_acc = valid_acc
            patience_counter = 0
    
        else:
            if args.early_stop:
                patience_counter += 1

        if patience_counter > early_stopping_patience:
            print("Early stopping triggered. Training stopped.")
            print('epoch', epoch_count, '| losses:', avg_loss)
            break
        print('epoch', epoch_count, '| losses:', avg_loss)
#     torch.optim.swa_utils.update_bn(train_dataloader, swa_model)
    run_eval(args, model, datasets, tokenizer, split='test',criterion=criterion)
    plot(all_train_acc, all_valid_acc)
    

def run_eval(args, model, datasets, tokenizer, split='validation', criterion=None, epoch_count=0, classifier=None):

    model.eval()
    if classifier: classifier.eval()
        
    dataloader = get_dataloader(args, datasets[split], split, drop_last=True)


    acc = []
    losses = []  

    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        
        inputs, labels = prepare_inputs(batch, model)
        
        if classifier:
            embed, cls_token = model(inputs, labels)
            logits = classifier(cls_token)
        elif args.task=="supcon":
            logits, cls_token = model(inputs, labels)
        else:
            logits = model(inputs, labels)
            
        if criterion is not None:
            if(args.cl_learning == "SimCLR") and not classifier: 
                loss = criterion(logits)
            else:
                loss = criterion(logits, labels)
                
            losses.append(loss.item())

        if args.task=="supcon" and classifier is None:
            pass
        else:
            tem = (logits.argmax(1) == labels).float().sum() / float(len(labels))
            acc.append(tem.item())
            
    ave_acc = -1 if len(acc) == 0 else np.mean(acc)
    avg_loss = -1 if len(losses) == 0 else np.mean(losses)
    
    
    log_message = ''
    if avg_loss is not None:
        log_message += f'{split} avg loss: {avg_loss} avg acc: {ave_acc}\n'

    if epoch_count % args.log_interval == 0 or epoch_count == args.n_epochs - 1:
        
        with open(args.file_name, 'a') as log_file:
            log_file.write(log_message)
    

    print(log_message)
    return ave_acc


def supcon_train(args, model, datasets, tokenizer):
    if args.early_stop:
        patience_counter = 0
        best_acc = 0
        early_stopping_patience = 2

    criterion = SupConLoss(temperature=args.temperature)

    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], drop_last=True)
    # task2: setup model's optimizer_scheduler if you have
    adam_opt = torch.optim.Adam(model.parameters(), lr=args.embed_lr, eps=args.adam_epsilon)
    # task3: write a training loop
    #model.load_state_dict(torch.load("ma.pth")) 
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in enumerate(progress_bar(train_dataloader)):
            
            inputs, labels = prepare_inputs(batch, model)
            features, _ = model(inputs, labels)
            
            if(args.cl_learning == "SimCLR"): 
                # both labels and mask are None -> SimCLR
                loss = criterion(features)
            elif(args.cl_learning == "SupCon"):
                loss = criterion(features, labels) 
            else:
                raise ValueError("supcon train must use SimCLR or SupCon loss") 
            losses += loss.item()
            loss.backward()
            adam_opt.step() 
            adam_opt.zero_grad()
#         torch.save(model.state_dict(), "mb.pth")
            
        avg_loss = losses / len(train_dataloader)
        run_eval(args, model, datasets, tokenizer, split='validation',criterion=criterion, epoch_count=epoch_count)
        if epoch_count % args.log_interval == 0 or epoch_count == args.n_epochs - 1:
            with open(args.file_name, 'a') as log_file:
                log_file.write(f'\nepoch: {epoch_count}, losses: {avg_loss}\n')
                
        print('epoch', epoch_count, '| losses:', avg_loss)

    criterion = nn.CrossEntropyLoss()
    classifier = Classifier(args, target_size=model.target_size).to(device)
    #cls.load_state_dict(torch.load("a.pth"))  #if want to train faster
    adam_opt = torch.optim.Adam(classifier.parameters(), lr=args.classify_LR)
    print("Start training Classifier...")
    for epoch_count in range(args.n_epochs_CL_CLS): #### train classifier
        losses = 0
        model.eval()
        classifier.train()
        acc=[]

        for step, batch in enumerate(progress_bar(train_dataloader)):
            
            inputs, labels = prepare_inputs(batch, model)
            _, cls_token = model(inputs, labels)
            logits = classifier(cls_token)

            loss = criterion(logits,labels)
            losses += loss.item()
            
            loss.backward()
            adam_opt.step() 
            adam_opt.zero_grad()
            
            tem = (logits.argmax(1) == labels).float().sum() / len(labels)
            acc.append(tem.item())

        avg_loss = losses / len(train_dataloader)
        ave_acc = np.mean(acc)
        
#         torch.save(classifier.state_dict(), "b.pth")
        
        if epoch_count % args.log_interval == 0 or epoch_count == args.n_epochs - 1:
            with open(args.file_name, 'a') as log_file:
                log_file.write(f'\nepoch: {epoch_count}, losses: {avg_loss}, acc: {ave_acc}\n')
                
        valid_acc = run_eval(args, model, datasets, tokenizer, split='validation',criterion=criterion, epoch_count=epoch_count, classifier=classifier)
        run_eval(args, model, datasets, tokenizer, split='test',criterion=criterion, epoch_count=epoch_count, classifier=classifier)
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            patience_counter = 0
    
        else:
            if args.early_stop:
                patience_counter += 1

        if patience_counter > early_stopping_patience:
            print("Early stopping triggered. Training stopped.")
            print('epoch', epoch_count, '| losses:', avg_loss,'acc: ', ave_acc)
            break
        print('epoch', epoch_count, '| losses:', avg_loss, 'acc: ', ave_acc)

def write_outputs(args):
    file_name = f"{args.save_dir}/{args.task}.txt"
    duplicate_count = 1
    while os.path.isfile(file_name):
        file_name = f"{args.save_dir}/{args.task}_{duplicate_count}.txt"
        duplicate_count += 1
    with open(file_name, 'a') as log_file:
        arguments = [
            f"batch-size: {args.batch_size}",
            f"learning-rate: {args.learning_rate}",
            f"hidden-dim: {args.hidden_dim}",
            f"drop-rate: {args.drop_rate}",
            f"embed-dim: {args.embed_dim}",
            f"adam-epsilon: {args.adam_epsilon}",
            f"n-epochs: {args.n_epochs}",
            f"max-len: {args.max_len}"
        ]
        header_line = ', '.join(arguments)
        log_file.write(header_line + '\n\n')
    args.file_name = file_name

if __name__ == "__main__":
    args = params()
    args = setup_gpus(args)
    args = check_directories(args)
    set_seed(args)

    cache_results, already_exist = check_cache(args)
    tokenizer = load_tokenizer(args)

    if already_exist:
        features = cache_results
    else:
        data = load_data()
        features = prepare_features(args, data, tokenizer, cache_results)
    datasets = process_data(args, features, tokenizer)
    for k,v in datasets.items():
        print(k, len(v))

    write_outputs(args)
    
    print(f'Output file: {args.file_name}')

    if args.task == 'baseline':
        model = IntentModel(args, tokenizer, target_size=60, datasets=datasets).to(device)
        baseline_train(args, model, datasets, tokenizer)

    elif args.task == 'custom': # you can have multiple custom task for different techniques
        model = CustomModel(args, tokenizer, target_size=60, datasets=datasets).to(device)
        custom_train(args, model, datasets, tokenizer)
        
    elif args.task == 'supcon':
        model = SupConModel(args, tokenizer, target_size=60).to(device)
        supcon_train(args, model, datasets, tokenizer)
   
