import config as cfg

import numpy as np # linear algebra

import torch

from transformers import (AutoTokenizer, 
                          AutoModelForTokenClassification, 
                          pipeline
                         )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NERModel: 
    def __init__(self): 
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.ner_preset)
        self.model = AutoModelForTokenClassification.from_pretrained(cfg.ner_preset).to(device)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="max", device = device)
    
    def __call__(self, batch): 
        return self.nlp(batch)