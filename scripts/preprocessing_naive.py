import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

def naive_preprocessing(scraped_content):
    sentences = sent_tokenize(scraped_content)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    cleaned_sentences = []
    
    for sentence in sentences:
        # tokenizing
        words = word_tokenize(sentence.lower())
        
        cleaned_words = []
        for word in words:
            # removing punctuation
            word = re.sub(r'[^\w\s]', '', word)
            
            # checking if word is valid after punctuation removal
            if (word and len(word) >= 2 and word.lower() not in stop_words and not word.isnumeric()):
                lemmatized_word = lemmatizer.lemmatize(word.lower())
                cleaned_words.append(lemmatized_word)
        
        cleaned_sentences.append(' '.join(cleaned_words))
    
    return sentences, cleaned_sentences


