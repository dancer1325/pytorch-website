#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

def get_pytorch_urls():
    url = "https://pytorch.org/"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all links
        links = soup.find_all('a', href=True)
        
        urls = set()
        for link in links:
            href = link['href']
            full_url = urljoin(url, href)
            
            # Only include pytorch.org URLs and exclude fragments/anchors
            if 'pytorch.org' in full_url and '#' not in full_url:
                urls.add(full_url)
        
        return sorted(urls)
        
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []

if __name__ == "__main__":
    urls = get_pytorch_urls()
    
    # Save to file
    with open('pytorch_urls.txt', 'w') as f:
        f.write(f"Found {len(urls)} URLs from pytorch.org:\n")
        f.write("-" * 50 + "\n")
        for url in urls:
            f.write(url + "\n")
    
    print(f"Saved {len(urls)} URLs to pytorch_urls.txt")
