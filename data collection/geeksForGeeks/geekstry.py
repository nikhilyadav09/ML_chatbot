import requests
from bs4 import BeautifulSoup
import json
import re

class GeeksFGScraper:
    def __init__(self, base_url='https://www.geeksforgeeks.org/ml-introduction-data-machine-learning/'):
        self.base_url = base_url
        self.visited_pages = set()
        self.scraped_content = {}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove extra whitespaces, newlines, and trim
        return re.sub(r'\s+', ' ', str(text)).strip()

    def extract_page_content(self, soup):
        """Extract content from the page"""
        content = {}

        # Extract main article title
        title = soup.find('h1', class_='entry-title')
        content['title'] = self.clean_text(title.get_text()) if title else 'Untitled'

        # Extract article sections
        sections = soup.find_all(['h2', 'h3'])
        for section in sections:
            section_title = self.clean_text(section.get_text())
            section_content = []

            # Find content for each section
            current = section.find_next_sibling()
            while current and current.name not in ['h2', 'h3']:
                # Extract text from paragraphs
                if current.name == 'p':
                    para_text = self.clean_text(current.get_text())
                    if para_text:
                        section_content.append(para_text)
                
                # Extract text from lists
                elif current.name in ['ul', 'ol']:
                    list_items = current.find_all('li')
                    for item in list_items:
                        item_text = self.clean_text(item.get_text())
                        if item_text:
                            section_content.append(item_text)
                
                current = current.find_next_sibling()

            if section_content:
                content[section_title] = section_content

        return content

    def find_next_article(self, soup):
        """Find link to next article using specific class"""
        # Find the div with class 'article-pgnavi_next'
        next_div = soup.find('div', class_='article-pgnavi_next')
        
        if next_div:
            # Find the anchor tag within this div
            next_link = next_div.find('a', class_='pg-head')
            
            # Extract href if link exists
            if next_link and next_link.has_attr('href'):
                return next_link['href']
        
        return None

    def scrape_page(self, url):
        """Scrape a single page"""
        if url in self.visited_pages:
            return None

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Mark as visited
            self.visited_pages.add(url)
            
            # Extract content
            page_content = self.extract_page_content(soup)
            
            # Find next article link
            next_article = self.find_next_article(soup)
            
            return {
                'content': page_content,
                'next_article': next_article
            }
        
        except requests.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return None

    def scrape_article_chain(self, start_url, max_articles=5):
        """Scrape a chain of related articles with limit"""
        current_url = start_url
        article_count = 0
        
        while current_url and article_count < max_articles:
            print(f"Scraping article: {current_url}")
            result = self.scrape_page(current_url)
            
            if result:
                # Store content
                self.scraped_content[current_url] = result['content']
                
                # Move to next article
                current_url = result['next_article']
                article_count += 1
            else:
                break

    def save_to_json(self, filename='geeksforgeeks_ml_content5.json'):
        """Save scraped content to a JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_content, f, ensure_ascii=False, indent=4)
        print(f"Content saved to {filename}")

def main():
    # Starting URL for Machine Learning article
    start_url = 'https://www.geeksforgeeks.org/introduction-deep-learning/'
    
    scraper = GeeksFGScraper()
    scraper.scrape_article_chain(start_url)
    scraper.save_to_json()

if __name__ == '__main__':
    main()