#!/usr/bin/env python3
from nltk.tokenize import word_tokenize
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import gensim
from pymystem3 import Mystem


class TextPreprocessor:
    def __init__(self, lang):
        """Initialize TextPreprocessor
        
        Arguments
        ---------
            lang (str): text language ('eng'/'rus')
        """
        if lang == 'rus':
            nltk.download('wordnet')
            
            self.stop_words = stopwords.words("russian")
            self.lemmatizer = Mystem()
        else:
            self.stop_words = gensim.parsing.preprocessing.STOPWORDS
            self.lemmatizer = WordNetLemmatizer()
    
    def lower_topic(self, topic):
        """Return the lowercased strings."""    
        try: 
            return topic.lower()
        except AttributeError:
            return topic 

    def remove_ext_spaces(self, topic):
        """Remove extra spaces"""
        try:
            return " ".join(topic.split())
        except AttributeError:
            return topic

    def remove_punct(self, topic):
        """Remove punctuation"""
        try:
            return topic.translate(str.maketrans(' ', ' ', string.punctuation))
        except AttributeError:
            return topic

    def tokenize(self, topic):    
        try:
            return word_tokenize(topic)
        except (AttributeError, TypeError):
            return topic

    def remove_stop_words(self, topic):
        """Remove stop-words"""
        try:
            return [w for w in topic if not w in self.stop_words]
        except (AttributeError, TypeError):
            return topic

    def lemmatize(self, topic):
        try:
            return " ".join([self.lemmatizer.lemmatize(w) for w in topic])
        except (AttributeError, TypeError):
            return topic   