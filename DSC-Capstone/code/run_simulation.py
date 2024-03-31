import torch
import pickle
import configs
import argparse
import os
import sys
from LSTM_simulate import LSTM_simulate
from cnn_simulate import cnn_simulate
from dnn_simulate import dnn_simulate
from transformer_simulate import transformer_simulate

def main(args):
    print("GPU availability: {}".format(torch.cuda.is_available()))
    GPU = False
    other = False

    if args.gpu=="True" and torch.cuda.is_available():
        GPU = True
    elif args.gpu=="True":
        print("GPU is not available on this environment")
    if args.other=="True":
        other = True
    env = args.env
    trial = args.trial
    path = "profiling/"+env+"/trial_" + trial +"/"
    
    

    if other:
        os.makedirs(path+"other/CNN")
        cnn_simulate(env,"other", trial)
        os.makedirs(path+"other/DNN")
        dnn_simulate(env,"other", trial)
        os.makedirs(path+"other/LSTM")
        try:   
            LSTM_simulate(env,"other", trial)
        except:
            print("Skip LSTM, cuda version issue")
        os.makedirs(path+"other/transformer")
        transformer_simulate(env,"other", trial)
    else:
        os.makedirs(path+"cpu/CNN")
        cnn_simulate(env,"cpu", trial)
        os.makedirs(path+"cpu/DNN")
        dnn_simulate(env,"cpu", trial)
        os.makedirs(path+"cpu/LSTM")
        try:
            LSTM_simulate(env,"cpu", trial)
        except:
            pass
        os.makedirs(path+"cpu/transformer")
        transformer_simulate(env,"cpu", trial)



    if GPU:
        os.makedirs(path+"gpu/CNN")
        cnn_simulate(env,"gpu", trial)
        os.makedirs(path+"gpu/DNN")
        dnn_simulate(env,"gpu", trial)
        os.makedirs(path+"gpu/LSTM")
        try:
            LSTM_simulate(env,"gpu", trial)
        except:
            pass
        os.makedirs(path+"gpu/transformer")
        transformer_simulate(env,"gpu", trial)



if __name__ == "__main__":
    #Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--other', type=str)
    parser.add_argument('--trial', type=str)
    
    args = parser.parse_args()
    main(args)
