import torch
import torch.nn as nn


class Experiment_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_setup()
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        
    def model_setup(self):
#         print(f"Setting up {args.model} model")
        model_id = "google/gemma-2b-it"
        os.environ["HF_TOKEN"] = "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"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.8)
        
    def forward(self):
        
