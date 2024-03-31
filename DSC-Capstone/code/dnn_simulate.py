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
import timeit

def dnn_simulate(env, dev, trial):
    device = torch.device("cpu")
    GPU = False
    GPU2 = False
    if dev == "gpu":
        device = torch.device("cuda")
        GPU = True
    if torch.cuda.is_available():
        GPU2 = True    
        
    model = model_util.DNN(300, 52, 25).to(device)
    if GPU:
        model.load_state_dict(torch.load("./models/dnn_model.pth")) 
    else:
        model.load_state_dict(torch.load("./models/dnn_model.pth",map_location=torch.device('cpu')))   

    with open ('small_sample', 'rb') as fp:
        data_feature = pickle.load(fp)
    fp.close()
    model.eval()

    act = [ProfilerActivity.CPU]
    if GPU:
        act.append(ProfilerActivity.CUDA)

    with profile(activities=act, profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            #################### Profiling this part##########################
            for i in range(10):
                input = torch.Tensor([data_feature[i]]).to(device)
                output = model(input)


                pred = torch.max(output, 1)[1]
                print("Activity ID: ", pred.item())
            ##################################################################

    environment_name = env
    rpath = "./profiling/"+ environment_name +"/trial_"+trial+"/cpu/DNN/"
    if GPU:
        rpath = "./profiling/"+ environment_name +"/trial_"+trial+"/gpu/DNN/"
    elif dev=="other":
        rpath = "./profiling/"+ environment_name +"/trial_"+trial+"/other/DNN/"

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
        len_list = [50, 100, 500, 1000, 5000]
        for i in len_list:
            torch.manual_seed(0)
            large_data = torch.rand(i, 300, 52).to(device)  
            try:
                starttime = timeit.default_timer()
                model(large_data)
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

