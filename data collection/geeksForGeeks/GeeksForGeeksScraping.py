import requests
from bs4 import BeautifulSoup
import json

def scrape_geeksforgeeks_ml_article():
    # Initial URL to start scraping
    base_url = "https://www.geeksforgeeks.org/what-is-machine-learning/"
    
    # Dictionary to store content by topics
    content_dict = {
        "Introduction": [],
        "Definition": [],
        "Advantages": [],
        "Disadvantages": [],
        "Conclusion": [],
        "FAQs": []
    }
    
    # Set to keep track of visited URLs to prevent infinite loops
    visited_urls = set()
    
    def extract_text_from_page(url):
        # Prevent revisiting the same URL
        if url in visited_urls:
            return None
        visited_urls.add(url)
        
        try:
            # Send a request to the webpage
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            # Check if the request was successful
            if response.status_code != 200:
                print(f"Failed to retrieve {url}")
                return None
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract content from different sections
            content_dict["Introduction"].extend([p.get_text(strip=True) for p in soup.find_all('p', dir="ltr") if p.find('span') and 'Machine learning is' in p.get_text()])
            
            # Definition section
            definition_sections = soup.find_all('h2', string='What is Machine Learning?')
            if definition_sections:
                next_paragraphs = definition_sections[0].find_next_siblings('p')
                content_dict["Definition"].extend([p.get_text(strip=True) for p in next_paragraphs])
            
            # Advantages section
            advantages_sections = soup.find_all('h2', string='Advantages of Machine Learning')
            if advantages_sections:
                advantages = advantages_sections[0].find_next_siblings('h3')
                content_dict["Advantages"].extend([sect.get_text(strip=True) for sect in advantages])
            
            # Disadvantages section
            disadvantages_sections = soup.find_all('h2', string='Disadvantages of Machine Learning')
            if disadvantages_sections:
                disadvantages = disadvantages_sections[0].find_next_siblings('h3')
                content_dict["Disadvantages"].extend([sect.get_text(strip=True) for sect in disadvantages])
            
            # Conclusion section
            conclusion_sections = soup.find_all('h2', string='Conclusion')
            if conclusion_sections:
                next_paragraphs = conclusion_sections[0].find_next_siblings('p')
                content_dict["Conclusion"].extend([p.get_text(strip=True) for p in next_paragraphs])
            
            # FAQs section
            faqs_sections = soup.find_all('h2', string='Advantages and Disadvantages of Machine Learning-FAQs')
            if faqs_sections:
                faq_questions = faqs_sections[0].find_next_siblings(['h3', 'blockquote'])
                content_dict["FAQs"].extend([sect.get_text(strip=True) for sect in faq_questions])
            
            # Find next page link
            next_page_links = soup.find_all('a', class_='pg-head', href=True)
            next_page = [link['href'] for link in next_page_links if 'next_article' in link.get_text()]
            
            return next_page[0] if next_page else None
        
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None
    
    # Start scraping from the base URL
    current_url = base_url
    while current_url:
        current_url = extract_text_from_page(current_url)
    
    return content_dict

def main():
    # Scrape the article
    ml_content = scrape_geeksforgeeks_ml_article()
    
    # Save content to a JSON file
    with open('machine_learning_article.json', 'w', encoding='utf-8') as f:
        json.dump(ml_content, f, indent=2, ensure_ascii=False)
    
    # Print summary of scraped content
    for topic, content in ml_content.items():
        print(f"\n{topic}:")
        for item in content[:3]:  # Print first 3 items for each topic
            print(f"- {item}")
        print(f"Total items in {topic}: {len(content)}")

if __name__ == "__main__":
    main()