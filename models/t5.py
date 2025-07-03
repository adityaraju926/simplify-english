from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')

def summary(input_text, tokenizer, model, config, input_length):
    inputs = tokenizer(input_text, return_tensors="pt", max=input_length, truncation=True, padding="max")
    
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max=config["max"],
        min=config["min"],
        num_beams=config["beams"],
        early_stopping=True,
        length_threshold=config["length_threshold"],
        repetition=config["repetition"]
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def paragraph_chunking(text, tokenizer, max_tokens=750):
    paragraphs = re.split(r'\n\s*\n', text)
    
    clean_paragraphs = []
    for paragraph in paragraphs:
        if paragraph.strip():
            clean_paragraphs.append(paragraph.strip())
    paragraphs = clean_paragraphs
    
    if len(paragraphs)<=1:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        clean_sentences = []
        for sentence in sentences:
            if sentence.strip():
                clean_sentences.append(sentence.strip())
        paragraphs = clean_sentences
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        initial_chunk = current_chunk + " " + paragraph if current_chunk else paragraph
        if len(tokenizer.encode(initial_chunk))>max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(tokenizer.encode(paragraph))>max_tokens:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                temporary = ""
                for sentence in sentences:
                    sentence_chunk_testing = temporary + " " + sentence if temporary else sentence
                    if len(tokenizer.encode(sentence_chunk_testing)) > max_tokens:
                        if temporary:
                            chunks.append(temporary.strip())
                        temporary = sentence
                    else:
                        temporary = sentence_chunk_testing
                if temporary:
                    current_chunk = temporary
            else:
                current_chunk = paragraph
        else:
            current_chunk = initial_chunk
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def main(cleaned_text, english_level):
    english_level_configs = {
        "beginner": {
            "max": 600, "min": 100, "beams": 3,
            "length_threshold": 1.2, "repetition": 1.1, "temperature": 1.0,
            "prompt": "summarize in simple terms:"
        },
        "intermediate": {
            "max": 800, "min": 150, "beams": 4,
            "length_threshold": 1.0, "repetition": 1.3, "temperature": 1.0,
            "prompt": "summarize:"
        },
        "advanced": {
            "max": 1000, "min": 200, "beams": 5,
            "length_threshold": 0.8, "repetition": 1.5, "temperature": 0.9,
            "prompt": "summarize comprehensively:"
        }
    }
    
    config_value = english_level_configs[english_level.lower()]
    input_length = 512
    
    chunks = paragraph_chunking(cleaned_text, tokenizer, max_tokens=input_length)
    
    summary_chunks = []
    for chunk in chunks:
        input_text = config_value["prompt"] + chunk
        summary = summary(input_text, tokenizer, model, config_value, input_length)
        summary_chunks.append(summary)

    combined_summary_input = " ".join(summary_chunks)
    final_prompt = f"{config_value['prompt']}the following text clearly" + combined_summary_input
    
    final_summary = summary(final_prompt, tokenizer, model, config_value, input_length)
    
    return final_summary


