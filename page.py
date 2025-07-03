import spacy
import subprocess
import sys

try:
    spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)

import streamlit as st
from scripts.scraping import article_content
from scripts.preprocessing_textrank import textrank_preprocessing
from models.textrank import textrank_summary

def summary(article_url, english_level):
    scraped_content = article_content(article_url)
    raw_sentences, preprocessed_sentences = textrank_preprocessing(scraped_content)
    summary = textrank_summary(raw_sentences, preprocessed_sentences, english_level.lower())
    
    st.text_area("TextRank Summary", summary, height=400)

def UI():
    column1, column2 = st.columns(2)
    with column1:
        article_url = st.text_input(label="Enter URL", placeholder="example.com")
    
    with column2:        
        english_level = st.selectbox(label="Choose English Level", options=["Select an option", "Beginner", "Intermediate", "Advanced"])
    
    st.write("")
    
    button_clicked = False
    if st.button("Create Summary", type="primary"):
        button_clicked = True
    
    return article_url, english_level, button_clicked

def main():
    st.title("Simplify English")
    article_url, english_level, button_clicked = UI()
    if button_clicked:
        summary(article_url, english_level)
        
if __name__ == "__main__":
    main()

