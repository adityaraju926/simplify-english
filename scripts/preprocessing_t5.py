import re

def t5_preprocessing(scraped_content):
    # removing website contents that aren't relevant to the article meaning
    text = re.sub(r'Photo by [^.]*\.', '', scraped_content)
    text = re.sub(r'\d+ min read', '', text)
    text = re.sub(r'ListenShare', '', text)
    text = re.sub(r'--\d+', '', text)
    
    # removing punctuation and spacing
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    # cleaning up the whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
