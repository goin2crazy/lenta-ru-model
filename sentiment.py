import .config as cfg


import numpy as np # linear algebra
import torch

from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          pipeline
                         )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentimentModel:     
    def __init__(self): 
        model_checkpoint = cfg.sentiment_preset
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint).to(device)

        
    def get_sentiment(self, text, focus = ''):
        """ Calculate sentiment of a text. `return_type` can be 'label', 'score' or 'proba' """
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
            outputs = self.model(**inputs).logits
            
            probas = torch.sigmoid(outputs).cpu().numpy()
    
        for proba in probas: 
              yield proba.dot([-1, 0, 1])
        
    def __call__(self, *args, **kwargs): 
        return self.get_sentiment( *args, **kwargs)