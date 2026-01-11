#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urlparse
import time

def sanitize_filename(url):
    """Convert URL to safe filename"""
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    if not path:
        path = 'index'
    # Replace special chars with underscores
    filename = re.sub(r'[^\w\-_.]', '_', path)
    return f"{filename}.md"

def extract_content(url):
    """Extract main content from webpage"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else url
        
        # Get main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        content = main_content.get_text(separator='\n', strip=True) if main_content else ""
        
        return title_text, content
        
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None, None

def scrape_pytorch_pages():
    # Read URLs from file
    with open('pytorch_urls.txt', 'r') as f:
        lines = f.readlines()
    
    urls = [line.strip() for line in lines if line.strip() and line.startswith('https://')]
    
    # Create output directory
    os.makedirs('pytorch_pages', exist_ok=True)
    
    for i, url in enumerate(urls, 1):
        print(f"Processing {i}/{len(urls)}: {url}")
        
        title, content = extract_content(url)
        
        if content:
            filename = sanitize_filename(url)
            filepath = os.path.join('pytorch_pages', filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n")
                f.write(f"**URL:** {url}\n\n")
                f.write("---\n\n")
                f.write(content)
            
            print(f"  Saved: {filename}")
        
        # Be nice to the server
        time.sleep(1)
    
    print(f"\nCompleted! Scraped {len(urls)} pages to 'pytorch_pages/' directory")

if __name__ == "__main__":
    scrape_pytorch_pages()
