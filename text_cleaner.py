# Save as text_helpers.py

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stopword_set = set(stopwords.words('english'))

def clean_text(text):
    # Convert to lowercase and strip leading/trailing spaces
    text = str(text).lower().strip()

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7f]', r'', text)

    # Remove whole numbers (tokens of digits)
    text = re.sub(r'\b\d+\b', '', text)

    # Replace multiple periods (like "..." or "....") with a single space
    text = re.sub(r'\.{2,}', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stopword_set]

    # Remove extra spaces and join tokens
    cleaned_text = ' '.join(cleaned_tokens).strip()

    return cleaned_text
