import .config as cfg 

from .ner import NERModel
from .sentiment import SentimentModel

class Inference: 
    def __init__(self): 
        self.ner_model = NERModel()
        self.sentiment_model = SentimentModel()

    def predict_sentiment(self, word, text): 
        """

        Args:
          word: focus word
          text: text to analize

        Returns:

        """
        p = list(self.sentiment_model(f"[focus: {word}] {text}"))

        return sum(p)/len(p)

    def add_sentiments(self, ner_text, text):
        text_sentences = text.split('.')

        ner_sentiments = {i['word']: self.predict_sentiment(i['word'], text) for i in ner_text}

        ner_sentiments = [{**nt, "sentiment": v} for nt, (k, v) in zip(ner_text, ner_sentiments.items())]
        return  ner_sentiments

    def process(self, batch):
        """
        Arguments: 
            batch: list is strs
        """
        
        ner_batch = self.ner_model(batch)
        
        return [self.add_sentiments(ner_text, text) for ner_text, text in zip(ner_batch, batch)] 
            
    def __call__(self, batch): 
        if type(batch) == str: 
            batch = [batch]

        return self.process(batch)
