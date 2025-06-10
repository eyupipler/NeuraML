import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QTextBrowser, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QUrl, pyqtSignal, QObject, QThread
from googlesearch import search
from bs4 import BeautifulSoup
import requests
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import spacy
import wikipedia
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

nlp = spacy.load("xx_ent_wiki_sm")
nlp.add_pipe("sentencizer")

class Worker(QObject):
    finished = pyqtSignal(str)

    def __init__(self, query):
        super().__init__()
        self.query = query

    def run(self):
        results_text = ""
        driver = None  # Selenium driver'ını yeniden kullanılabilir yapmak için
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
                'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7'
            }
            search_results = search(self.query, num=10, lang="en", stop=50, pause=2)
            search_results = list(search_results)

            for i, result in enumerate(search_results, 1):
                # Eğer sonuç wikipedia.org içeriyorsa doğrudan wikipedia kütüphanesini kullan
                if "wikipedia.org" in result:
                    try:
                        summary_text = wikipedia.summary(self.query, sentences=1)
                    except Exception as e:
                        summary_text = f"Wikipedia hatası: {e}"
                else:
                        try:
                            # Selenium driver'ı eğer henüz oluşturulmadıysa oluştur
                            if driver is None:
                                options = Options()
                                options.add_argument("--headless")
                                options.add_argument("--ignore-certificate-errors")
                                options.add_argument("--ignore-ssl-errors")
                                options.add_argument("--mute-audio")
                                driver = webdriver.Chrome(options=options)
                            driver.get(result)
                            try:
                                wait = WebDriverWait(driver, 3)
                                wait.until(EC.presence_of_element_located((By.XPATH, "//meta[@name='description']")))
                            except Exception:
                                pass
                            try:
                                meta_element = driver.find_element(By.XPATH, "//meta[@name='description']")
                                summary_content = meta_element.get_attribute("content")
                            except Exception:
                                summary_content = ""
                            summary_text = self.summarize_text(summary_content)
                        except Exception as e:
                            summary_text = f"Hata: {e}"
                results_text += f"{summary_text}\n"

        except Exception as e:
            results_text = f"Hata: {e}"

        if driver is not None:
            driver.quit()

        if results_text:
            with open('Logs/results_logs.txt', 'a') as f:
                f.write(f"{datetime.now()} - {self.query}\n{results_text}\n\n")

            with open('Logs/ml_logs.txt', 'a') as f:
                f.write(f"{datetime.now()} - {self.query}\n")

        self.finished.emit(results_text)

    @staticmethod
    def summarize_text(text):
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        sentences.sort(key=len, reverse=True)
        return sentences[0] if sentences else ""