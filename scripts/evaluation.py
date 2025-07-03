from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def evaluate_summaries():
    raw_text = open("../data/raw.txt", 'r', encoding='utf-8').read().strip()
    
    naive_beginner = open("../data/naive/summary_naive_beginner.txt", 'r', encoding='utf-8').read().strip()
    naive_intermediate = open("../data/naive/summary_naive_intermediate.txt", 'r', encoding='utf-8').read().strip()
    naive_advanced = open("../data/naive/summary_naive_advanced.txt", 'r', encoding='utf-8').read().strip()
    
    textrank_beginner = open("../data/textrank/summary_textrank_beginner.txt", 'r', encoding='utf-8').read().strip()
    textrank_intermediate = open("../data/textrank/summary_textrank_intermediate.txt", 'r', encoding='utf-8').read().strip()
    textrank_advanced = open("../data/textrank/summary_textrank_advanced.txt", 'r', encoding='utf-8').read().strip()
    
    t5_beginner = open("../data/t5/summary_t5_beginner.txt", 'r', encoding='utf-8').read().strip()
    t5_intermediate = open("../data/t5/summary_t5_intermediate.txt", 'r', encoding='utf-8').read().strip()
    t5_advanced = open("../data/t5/summary_t5_advanced.txt", 'r', encoding='utf-8').read().strip()
    
    # Encode all the text files including the raw text and summaries
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([raw_text, naive_beginner, naive_intermediate, naive_advanced, textrank_beginner, textrank_intermediate, textrank_advanced, t5_beginner, t5_intermediate, t5_advanced])
    
    naive_beginner_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    naive_intermediate_score = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
    naive_advanced_score = cosine_similarity([embeddings[0]], [embeddings[3]])[0][0]
    
    textrank_beginner_score = cosine_similarity([embeddings[0]], [embeddings[4]])[0][0]
    textrank_intermediate_score = cosine_similarity([embeddings[0]], [embeddings[5]])[0][0]
    textrank_advanced_score = cosine_similarity([embeddings[0]], [embeddings[6]])[0][0]
    
    t5_beginner_score = cosine_similarity([embeddings[0]], [embeddings[7]])[0][0]
    t5_intermediate_score = cosine_similarity([embeddings[0]], [embeddings[8]])[0][0]
    t5_advanced_score = cosine_similarity([embeddings[0]], [embeddings[9]])[0][0]
    
    # Calculating averages across all the summary files per model
    naive_average = (naive_beginner_score + naive_intermediate_score + naive_advanced_score)/3
    textrank_average = (textrank_beginner_score + textrank_intermediate_score + textrank_advanced_score)/3
    t5_average = (t5_beginner_score + t5_intermediate_score + t5_advanced_score)/3
    
    print("Scores:")
    print("Naive:", f"{naive_average:.2f}")
    print("TextRank:", f"{textrank_average:.2f}")
    print("T5:", f"{t5_average:.2f}")

if __name__ == "__main__":
    evaluate_summaries()
