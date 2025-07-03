import requests
from bs4 import BeautifulSoup
import re

def article_content(article_url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    response = requests.get(article_url, headers=headers, timeout=10)
    article_contents = BeautifulSoup(response.content, 'html.parser')
    
    for content in article_contents(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
        content.decompose()

    # picking out components that are useful to the meaning
    selected_contents = ['title', 'article', '[role="main"]', '.content', '.article-content', '.post-content', '.entry-content', 'main']
    
    total_contents = []
    
    for selected_content in selected_contents:
        contents = article_contents.select(selected_content)
        if contents:
            for content in contents:
                text = content.get_text().strip()
                if text:
                    total_contents.append(text)
    
    if total_contents:
        final_content = ' '.join(total_contents)
    else:
        final_content = article_contents.get_text()
    
    final_content = re.sub(r'\s+', ' ', final_content)
    final_content = final_content.strip()
    
    return final_content
