from sklearn.feature_extraction.text import TfidfVectorizer

def create_naive_summary(original_sentences, cleaned_sentences, english_level):
    english_level_configs = {
        "beginner": {"summary": 0.8, "max": 70, "max_df": 0.7},
        "intermediate": {"summary": 0.6, "max": 90, "max_df": 0.8},
        "advanced": {"summary": 0.4, "max": 150, "max_df": 0.9}
    }
    
    config_value = english_level_configs[english_level.lower()]
    
    vectors = TfidfVectorizer(max_features=config_value["max"], min_df=1, max_df=config_value["max_df"])
    vectors.fit_transform(cleaned_sentences)
    feature_names = vectors.get_feature_names_out()
    
    sentence_scores = []
    for index, cleaned_sentence in enumerate(cleaned_sentences):
        cleaned_sentence_words = cleaned_sentence.split()
        keyword_count = 0
        for feature in cleaned_sentence_words:
            if feature in feature_names:
                keyword_count+=1
        if len(cleaned_sentence_words)>0:
            score = keyword_count/len(cleaned_sentence_words)
        else:
            score = 0
        
        sentence_scores.append({'index': index, 'original_sentence': original_sentences[index], 'score': score})
    
    sentence_scores.sort(key=lambda x: x['score'], reverse=True)
    
    sentence_count = max(1, int(len(original_sentences)*config_value["summary"]))
    selected_sentences = sorted(sentence_scores[:sentence_count], key=lambda x: x['index'])
    
    summary_sentences = []
    for sentence in selected_sentences:
        summary_sentences.append(sentence['original_sentence'])
    
    summary_text = ' '.join(summary_sentences)
    
    return summary_text

 