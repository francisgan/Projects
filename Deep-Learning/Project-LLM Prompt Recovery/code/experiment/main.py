# from util import *
import torch
import os
from train import *
from model import *
import argparse

import pandas as pd
from transformers import pipeline, AutoTokenizer,GPT2Tokenizer, GPT2Model, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
# TODO determine which device to use (cuda or cpu)
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

batch_size = 4  # You can adjust this based on your requirements and GPU memory
class CustomDataset(Dataset):
    def __init__(self, batch_data, tokenizer):
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        
        for item in batch_data:
            input_text = f"Given the input:\n {item[0]}\n and the resulting output:\n {item[2]}.\n Determine the prompt used to generate the result: "
            tokenized_inputs = tokenizer(input_text, padding='max_length', max_length=200, truncation=True, return_tensors="pt")
            
            self.input_ids.append(tokenized_inputs['input_ids'].squeeze(0))  # Remove batch dimension
            self.attention_masks.append(tokenized_inputs['attention_mask'].squeeze(0))  # Remove batch dimension
            self.labels.append(item[1])  # Assuming labels are already in a suitable format (e.g., IDs, text)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }
    
if __name__ == "__main__":
    
    
    # Convert DataFrame to a numpy array for easier processing
    rewrite_data_df = pd.read_csv('../data/rewrite_data.csv')
    data = rewrite_data_df.values

    # First split: separate the test set
    train_val_data, test_data = train_test_split(data, test_size=0.15, random_state=42)

    # Second split: separate train and validation sets
    train_data, val_data = train_test_split(train_val_data, test_size=0.176, random_state=42)  # ~0.176 = 15/85
    
    train_dataset = CustomDataset(train_data, tokenizer)
    val_dataset = CustomDataset(val_data, tokenizer)
    test_dataset = CustomDataset(test_data, tokenizer)
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     for batch in train_loader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
        
    model = Experiment_Model(config, device)
#     losses, v_losses = train(model, data, data_val, char_idx_map, config, device)
    print(len(ds))
    print(ds[0])