import requests
import json
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote
import os
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QTextBrowser, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QUrl, pyqtSignal, QObject, QThread

# Yapılandırma
OUTPUT_FILE = "beyin_hastaliklari_verileri.jsonl"
HEADERS = {'User-Agent': 'MedikalVeriBot/1.0 (contact@example.com)'}

# Wikipedia API
WIKI_API = "https://tr.wikipedia.org/w/api.php"
WIKI_CATEGORY = "Kategori:Beyin_hastalıkları"

# PubMed API
PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUBMED_QUERY = '("beyin hastalıkları" OR "nörolojik bozukluklar") AND "turkish"[Language]'
PUBMED_PARAMS = {
    'db': 'pubmed',
    'retmax': 100,
    'email': 'your_email@example.com',
    'tool': 'MedicalDataExtractor'
}

class Wiki(QObject):
    finished = pyqtSignal(str)
    """Wikipedia'dan beyin hastalıkları ile ilgili sayfaları çeker"""
    def __init__(self):
        super().__init__()
        self.wiki_api = WIKI_API
        self.wiki_category = WIKI_CATEGORY
        self.headers = HEADERS

    def fetch_category_pages(self):
        return get_wiki_category_pages()

    def fetch_page_details(self, page_title):
        return get_wiki_page_details(page_title)
    def get_wiki_category_pages():
        """Wikipedia kategorisindeki sayfaları alır"""
        all_pages = []
        params = {
            'action': 'query',
            'list': 'categorymembers',
            'cmtitle': WIKI_CATEGORY,
            'cmlimit': 500,
            'format': 'json'
        }
    
        while True:
            try:
                response = requests.get(WIKI_API, params=params, headers=HEADERS)
                response.raise_for_status()
                data = response.json()
    
                if 'query' in data:
                    all_pages.extend(data['query']['categorymembers'])
    
                if 'continue' in data:
                    params['cmcontinue'] = data['continue']['cmcontinue']
                    time.sleep(0.5)
                else:
                    break
                
            except Exception as e:
                print(f"Wikipedia hatası: {e}")
                break
            
        return all_pages
    
    
    def get_wiki_page_details(page_title):
        """Wikipedia sayfa detaylarını alır"""
        params = {
            'action': 'query',
            'prop': 'extracts|info',
            'titles': page_title,
            'exintro': True,
            'explaintext': True,
            'inprop': 'url',
            'format': 'json'
        }
    
        try:
            response = requests.get(WIKI_API, params=params, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            page = next(iter(data['query']['pages'].values()), None)
    
            if page and 'missing' not in page:
                return {
                    'title': page['title'],
                    'summary': page.get('extract', ''),
                    'url': page.get('fullurl', f'https://tr.wikipedia.org/wiki/{quote(page_title.replace(" ", "_"))}'),
                    'source': 'Wikipedia',
                    'timestamp': datetime.now().isoformat()
                }
            return None
    
        except Exception as e:
            print(f"Wikipedia işleme hatası ({page_title}): {e}")
            return None
    
    
    def get_pubmed_articles():
        """PubMed'den Türkçe makaleleri alır"""
        # Makale ID'lerini ara
        search_params = {**PUBMED_PARAMS, 'term': PUBMED_QUERY, 'retmode': 'json'}
        try:
            response = requests.get(f"{PUBMED_API}esearch.fcgi", params=search_params)
            response.raise_for_status()
            data = response.json()
            id_list = data.get('esearchresult', {}).get('idlist', [])
    
            if not id_list:
                print("PubMed'de Türkçe makale bulunamadı")
                return []
    
            # Makale detaylarını al
            fetch_params = {**PUBMED_PARAMS, 'id': ','.join(id_list), 'retmode': 'xml'}
            response = requests.get(f"{PUBMED_API}efetch.fcgi", params=fetch_params)
            response.raise_for_status()
    
            articles = []
            root = ET.fromstring(response.content)
    
            for article in root.findall('.//PubmedArticle'):
                # Başlık
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else "Başlık Yok"
    
                # Özet
                abstract = ""
                abstract_elem = article.find('.//AbstractText')
                if abstract_elem is not None:
                    abstract = ''.join(abstract_elem.itertext()).strip()
    
                # PMID ve URL
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else ""
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
    
                # Tarih
                date_elem = article.find('.//PubDate/Year')
                date = date_elem.text if date_elem is not None else "Tarih Yok"
    
                if abstract:
                    articles.append({
                        'title': f"{title} ({date})",
                        'summary': abstract,
                        'url': url,
                        'source': 'PubMed',
                        'timestamp': datetime.now().isoformat()
                    })
    
            return articles
    
        except Exception as e:
            print(f"PubMed hatası: {e}")
            return []
    
    
    def save_to_jsonl(data, filename):
        """Verileri JSONL dosyasına ekler"""
        mode = 'a' if os.path.exists(filename) else 'w'
        with open(filename, mode, encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return len(data)
    
    
    # Ana işlem
    if __name__ == "__main__":
        print("Beyin hastalıkları veri toplama başlatıldı...")
    
        # 1. Wikipedia verilerini topla
        print("\nWikipedia verileri alınıyor...")
        wiki_pages = get_wiki_category_pages()
        wiki_data = []
    
        for member in wiki_pages:
            if member['ns'] == 0:  # Sadece ana sayfalar
                page_data = get_wiki_page_details(member['title'])
                if page_data and page_data.get('summary'):
                    wiki_data.append(page_data)
                    print(f"Wikipedia: {member['title']}")
                time.sleep(0.3)
    
        # 2. PubMed verilerini topla
        print("\nPubMed verileri alınıyor...")
        pubmed_data = get_pubmed_articles()
        for article in pubmed_data:
            print(f"PubMed: {article['title']}")
    
        # 3. Tüm verileri kaydet
        total_saved = 0
        if wiki_data:
            saved = save_to_jsonl(wiki_data, OUTPUT_FILE)
            total_saved += saved
            print(f"\nWikipedia'dan {saved} kayıt eklendi")
    
        if pubmed_data:
            saved = save_to_jsonl(pubmed_data, OUTPUT_FILE)
            total_saved += saved
            print(f"PubMed'den {saved} kayıt eklendi")
    
        print(f"\nToplam {total_saved} yeni kayıt eklendi")
        print(f"Çıktı dosyası: {OUTPUT_FILE}")
        print("İşlem tamamlandı!")