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


def LSTM_simulate(env, dev, trial):
    device = torch.device("cpu")
    GPU = False
    GPU2 = False
    if dev == "gpu":
        device = torch.device("cuda")
        GPU = True
    if torch.cuda.is_available():
        GPU2 = True
    
 
    LSTM = model_util.LSTMClassifier(52, 250, 25, 2).to(device)
    if GPU:
        LSTM.load_state_dict(torch.load("./models/LSTM.pth")) 
    else:
        LSTM.load_state_dict(torch.load("./models/LSTM.pth",map_location=torch.device('cpu'))) 

    with open ('small_sample', 'rb') as fp:
        data_feature = pickle.load(fp)
    fp.close()
    LSTM.eval()

    act = [ProfilerActivity.CPU]
    if GPU:
        act.append(ProfilerActivity.CUDA)

    with profile(activities=act, profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            #################### Profiling this part##########################
            for i in range(10):
                input = torch.Tensor([data_feature[i]]).to(device)
                output = LSTM(input)

                pred = torch.max(output, 1)[1]
                print("Activity ID: ", pred.item())
            ##################################################################

    environment_name = env
    rpath = "./profiling/"+ environment_name +"/trial_"+trial+"/cpu/LSTM/"
    if GPU:
        rpath = "./profiling/"+ environment_name +"/trial_"+trial+"/gpu/LSTM/"
    elif dev=="other":
        rpath = "./profiling/"+ environment_name +"/trial_"+trial+"/other/LSTM/"

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
        len_list = [50, 100, 500, 1000, 2000]
        for i in len_list:
            torch.manual_seed(0)
            large_data = torch.rand(i, 300, 52).to(device)  
            try:
                starttime = timeit.default_timer()
                LSTM(large_data)
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

    

    


