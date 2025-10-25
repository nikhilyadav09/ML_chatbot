import requests
from bs4 import BeautifulSoup
import os
import json

class MLDocsScraper:
    def __init__(self, base_url='https://machinelearning101.readthedocs.io/en/latest/_pages/'):
        self.base_url = base_url
        self.visited_pages = set()
        self.scraped_content = {}

    def sanitize_filename(self, filename):
        """Remove invalid characters from filename."""
        print("success sanitization")
        return ''.join(c for c in filename if c.isalnum() or c in ['-', '_']).rstrip()

    def extract_text_by_topic(self, soup):
        """Extract text from different sections of the HTML."""
        content = {}

        # Extract main section headings and their content
        sections = soup.find_all(['h1', 'h2', 'h3'])
        for section in sections:
            section_title = section.get_text(strip=True)
            section_content = []

            # Find next siblings until next header
            current = section.find_next_sibling()
            while current and current.name not in ['h1', 'h2', 'h3']:
                # Extract text from paragraphs, blockquotes, etc.
                if current.name in ['p', 'blockquote', 'ol', 'ul']:
                    section_content.append(current.get_text(strip=True))
                current = current.find_next_sibling()

            content[section_title] = section_content

        return content

    def scrape_page(self, page_name):
        """Scrape a single page and extract content."""
        if page_name in self.visited_pages:
            return None

        url = self.base_url + page_name
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Mark as visited
            self.visited_pages.add(page_name)
            
            # Extract content
            page_content = self.extract_text_by_topic(soup)
            
            # Find next page link
            next_link = soup.find('a', {'accesskey': 'n'})
            next_page = next_link['href'] if next_link else None
            
            return {
                'content': page_content,
                'next_page': next_page
            }
        
        except requests.RequestException as e:
            print(f"Error scraping {page_name}: {e}")
            return None

    def scrape_all_pages(self, start_page='01_introduction.html'):
        """Scrape all pages in the documentation."""
        current_page = start_page
        
        while current_page:
            print(f"Scraping page: {current_page}")
            result = self.scrape_page(current_page)
            
            if result:
                # Store content
                self.scraped_content[current_page] = result['content']
                
                # Move to next page
                current_page = result['next_page']
            else:
                break

    def save_to_json(self, filename='ml_docs_content.json'):
        """Save scraped content to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_content, f, ensure_ascii=False, indent=4)
        print(f"Content saved to {filename}")

def main():
    scraper = MLDocsScraper()
    scraper.scrape_all_pages()
    scraper.save_to_json()

if __name__ == '__main__':
    main()