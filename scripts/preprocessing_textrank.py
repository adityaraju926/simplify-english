import spacy

nlp = spacy.load("en_core_web_sm")

def textrank_preprocessing(scraped_content):
    raw_text = scraped_content
    
    article = nlp(raw_text)
    
    original_sentences = []
    for sentence in article.sents:
        original_sentences.append(str(sentence))
    cleaned_sentences = []
    
    for sentence in article.sents:
        cleaned_words = []
        for token in sentence:
            # lemmatizing based on conditions 
            if (not token.is_stop and not token.is_punct and not token.is_space and not token.like_num and len(token.lemma_)>=2):
                cleaned_words.append(token.lemma_.lower())
        
        cleaned_sentences.append(' '.join(cleaned_words))
    
    return original_sentences, cleaned_sentences


