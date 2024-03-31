import argparse
import os

def params():
    parser = argparse.ArgumentParser()

    # Experiment options
    parser.add_argument("--task", default="baseline", type=str,\
                help="baseline is fine-tuning bert for classification;\n\
                      tune is advanced techiques to fine-tune bert;\n\
                      constast is contrastive learning method")
                      #choices=['baseline','tune','supcon'])
    parser.add_argument("--temperature", default=0.7, type=float, 
                help="temperature parameter for contrastive loss")

    # optional fine-tuning techiques parameters
    parser.add_argument("--reinit_n_layers", default=0, type=int, 
                help="number of layers that are reinitialized. Count from last to first.")
    
    # Others
    parser.add_argument("--input-dir", default='assets', type=str, 
                help="The input training data file (a text file).")
    parser.add_argument("--output-dir", default='results', type=str,
                help="Output directory where the model predictions and checkpoints are written.")
    parser.add_argument("--model", default='bert', type=str,
                help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", default="amazon", type=str,
                help="dataset", choices=['amazon'])
    

    # Key settings
    parser.add_argument("--ignore-cache", action="store_true",
                help="Whether to ignore cache and create a new input data")
    parser.add_argument("--debug", action="store_true",
                help="Whether to run in debug mode which is exponentially faster")
    parser.add_argument("--do-train", action="store_true",
                help="Whether to run training.")
    parser.add_argument("--do-eval", action="store_true",
                help="Whether to run eval on the dev set.")
    parser.add_argument("--early_stop", default=True, type=bool,
                help="Whether to do early stop")
    parser.add_argument("--advanced", default=[], type=int, action='append',
                    help="whether to use advanced technologies")
    
    # Hyper-parameters for tuning
    parser.add_argument("--batch-size", default=128, type=int,
                help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--learning-rate", default=1e-4, type=float, #0.000005
                help="Model learning rate starting point.")
    parser.add_argument("--hidden-dim", default=50, type=int,
                help="Model hidden dimension.")
    parser.add_argument("--drop-rate", default=0.8, type=float,
                help="Dropout rate for model training")
    parser.add_argument("--embed-dim", default=768, type=int,
                help="The embedding dimension of pretrained LM.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--n-epochs", default=10, type=int,
                help="Total number of training epochs to perform.")
    parser.add_argument("--max-len", default=20, type=int,
                help="maximum sequence length to look back")
    
    parser.add_argument("--log-interval", default=1, type=int,
                help="Interval of logging")

    # SupCon argument
    parser.add_argument("--cl-learning", default=None, type=str,
                help="set to SimCLR to use the SimCLR Loss, SupCon to use the SupCon loss")
    parser.add_argument("--embed-lr", default=0.000005, type=float,
                help="Learning rate for training encoder with Contrastive Learning")
    parser.add_argument("--CL-hidden-size", default=768, type=int,
                help="The dimension of hidden layer in MLP for CL")
    parser.add_argument("--embeded-size", default=128, type=int,
                help="The dimension of the word embedding after CL")
    
    parser.add_argument("--classify-LR", default=0.0001, type=float,
                help="Learning rate for training classifier with Contrastive Learning")
    parser.add_argument("--n-epochs-CL-CLS", default=50, type=int,
                help="Total number of  epochs to for training classifier with Contrastive Learning")
    
    parser.add_argument("--CL-Mask", default=1, type=int,
                help="Contrastive Learning: 1 turn on mask/ 0 turn off mask")

    # SWA and LLRD argument
    parser.add_argument("--swa_lr", default=1e-4, type=int,
            help="swa learning rate")
    parser.add_argument("--swa_start", default=1, type=int,
            help="when to start swa")
    parser.add_argument("--lr_mult", default=0.95, type=float,
            help="how much learning rate across layer")

    args = parser.parse_args()
    return args
