import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import networkx as nx


nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

lemmatizer = WordNetLemmatizer()
stopword_set = set(stopwords.words('english'))

import unicodedata

def normalize_whitespace(text):
    # Replace various Unicode space characters with normal space
    return re.sub(r'\s+', ' ', ''.join(
        ' ' if unicodedata.category(c).startswith('Z') else c for c in text
    )).strip()

def clean_text(text):
    # Ensure text is string, lowercase, and stripped
    text = str(text).lower().strip()

    # Normalize weird unicode whitespace before doing anything
    text = normalize_whitespace(text)

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove numbers
    text = re.sub(r'\d+', ' ', text)

    # Replace multiple periods with space
    text = re.sub(r'\.{2,}', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Normalize whitespace again before tokenizing
    text = normalize_whitespace(text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stopword_set and tok.strip()
    ]

    # Join tokens and normalize whitespace one last time
    cleaned_text = normalize_whitespace(' '.join(cleaned_tokens))

    return cleaned_text

