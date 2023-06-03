import wikipediaapi
import requests
from bs4 import BeautifulSoup
import re

def format_name(name):
    # Format the name with various separators
    name_variations = [name.replace(' ', sep) for sep in ['_', '-']]
    name_variations.extend([name.lower().replace(' ', sep) for sep in ['_', '-']])
    name_variations.extend([name.title().replace(' ', sep) for sep in ['_', '-']])

    return name_variations

def fetch_info(name, lang='en'):
    # Initialize the Wikipedia object with the specified language
    wiki_wiki = wikipediaapi.Wikipedia(lang)

    name_variations = format_name(name)
    for variation in name_variations:
        page_py = wiki_wiki.page(variation)
        if page_py.exists():
            return page_py.text
    return "None"

def get_bird_info(bird_name):
    name_variations = format_name(bird_name)
    language_code = 'en'  # Установите код языка только для английской версии

    for variation in name_variations:
        url = f'https://{language_code}.wikipedia.org/wiki/{variation.replace(" ", "_")}'
        print("Searching for:", url)
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Найти первый параграф
            p_tags = soup.find_all('p')
            for tag in p_tags:
                if tag.text and len(tag.text) > 50:
                    # Удалить скобки для цитирования и вернуть
                    return re.sub(r'\[[^\]]*\]', '', tag.text)

        except:
            pass
    return 'None'

