import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision
from typing import Optional, Union
import pickle
import model_util
import configs
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
import timeit


def transformer_simulate(env, dev, trial):
    device = torch.device("cpu")
    GPU = False
    GPU2 = False
    if dev == "gpu":
        device = torch.device("cuda")
        GPU = True
    if torch.cuda.is_available():
        GPU2 = True
    
    input_dim = 52  # Number of features per data point
    d_model = 256  # Size of the Transformer embeddings
    num_classes = 25  # Number of output classes
    num_heads = 8  # Number of heads in the multi-head attention mechanism
    num_layers = 3  # Number of Transformer layers
    max_seq_len = 300  # Maximum sequence length

    TF = model_util.TransformerClassifier(input_dim, d_model, num_classes, num_heads, num_layers, max_seq_len).to(device)
    if GPU:
        TF.load_state_dict(torch.load("./models/transformer.pth")) 
    else:
        TF.load_state_dict(torch.load("./models/transformer.pth",map_location=torch.device('cpu'))) 

    with open ('small_sample', 'rb') as fp:
        data_feature = pickle.load(fp)
    fp.close()
    TF.eval()

    act = [ProfilerActivity.CPU]
    if GPU:
        act.append(ProfilerActivity.CUDA)

    with profile(activities=act, profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            #################### Profiling this part##########################
            for i in range(10):
                input = torch.Tensor([data_feature[i]]).to(device)
                output = TF(input)

                pred = torch.max(output, 1)[1]
                print("Activity ID: ", pred.item())
            ##################################################################

    environment_name = env
    rpath = "./profiling/"+ environment_name +"/trial_"+trial+"/cpu/transformer/"
    if GPU:
        rpath = "./profiling/"+ environment_name +"/trial_"+trial+"/gpu/transformer/"
    elif dev=="other":
        rpath = "./profiling/"+ environment_name +"/trial_"+trial+"/other/transformer/"

    txt = prof.key_averages().table(sort_by="cpu_time_total")
    path = rpath +"time.txt"
    text_file = open(path, "w")
    text_file.write(txt)
    text_file.close()

    txt = prof.key_averages(group_by_input_shape=True).table()
    path = rpath +"shape.txt"
    text_file = open(path, "w")
    text_file.write(txt)
    text_file.close()

    path = rpath +"chromeTrace.json"
    prof.export_chrome_trace(path)

    if GPU2:
        txt = ""
        path = rpath +"large_sample_time.txt"
        len_list = [50, 100, 500, 1000]
        for i in len_list:
            torch.manual_seed(0)
            large_data = torch.rand(i, 300, 52).to(device)  
            try:
                starttime = timeit.default_timer()
                TF(large_data)
                endtime = timeit.default_timer()
                record = f"{i} samples, run time: {endtime - starttime}\n"
            except:
                record = f"{i} samples: failed\n"
                txt += record
                break
            txt += record
        text_file = open(path, "w")
        text_file.write(txt)
        text_file.close()

    


