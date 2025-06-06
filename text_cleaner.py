# Save as text_helpers.py

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stopword_set = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower().strip()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\.{2,}', ' ', text)
    tokens = word_tokenize(text)
    cleaned_tok = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stopword_set and tok not in string.punctuation]
    return ' '.join(cleaned_tok)
