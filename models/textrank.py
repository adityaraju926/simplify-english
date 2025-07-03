from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def textrank_summary(raw_sentences, preprocessed_sentences, english_level):
    english_level_configs = {
        "beginner": {"summary_ratio": 0.6, "max_features": 600, "max_df": 0.6, "probability": 0.9},
        "intermediate": {"summary_ratio": 0.4, "max_features": 1200, "max_df": 0.8, "probability": 0.8},
        "advanced": {"summary_ratio": 0.2, "max_features": 1500, "max_df": 0.9, "probability": 0.7}
    }
    
    config_value = english_level_configs[english_level.lower()]
    
    # Creating vectors and comparing similarity
    vectors = TfidfVectorizer(max_features=config_value["max_features"], min_df=1, max_df=config_value["max_df"])
    matrix = vectors.fit_transform(preprocessed_sentences)
    comparison = cosine_similarity(matrix)
    
    # Use PageRank algorithm
    scores = pagerank_algorithm(comparison, probability=config_value["probability"])
    
    sentence_scores = []
    for i, score in enumerate(scores):
        sentence_scores.append({'index': i, 'raw_sentence': raw_sentences[i], 'score': score})

    sentence_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Pick top sentences
    sentence_count = max(1, int(len(raw_sentences)*config_value["summary_ratio"]))
    top_sentences = sorted(sentence_scores[:sentence_count], key=lambda x: x['index'])
    
    summary_sentences = []
    for sentence in top_sentences:
        summary_sentences.append(sentence['raw_sentence'])
    
    summary_text = ' '.join(summary_sentences)
    
    return summary_text

def pagerank_algorithm(comparison, probability=0.85, iterations=100, difference=0.000001):
    n = len(comparison)
    scores = np.ones(n)/n
    
    comparison = comparison+1e-8
    sum = comparison.sum(axis=1, keepdims=True)

    final_matrix = comparison/sum
    final_matrix = probability*final_matrix+(1-probability)/n
    
    # Goes through the iterations and tweaks the scores
    for _ in range(iterations):
        new_scores = np.matmul(final_matrix.T, scores)
        if np.linalg.norm(new_scores-scores)<difference:
            break
        scores = new_scores
    
    return scores


