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

nlp = spacy.load("xx_ent_wiki_sm")
nlp.add_pipe("sentencizer")


class Worker(QObject):
    finished = pyqtSignal(str)

    def __init__(self, query):
        super().__init__()
        self.query = query

    def run(self):
        results_text = ""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
            search_results = search(self.query, num=10, lang="tr")

            for i, result in enumerate(search_results, 1):
                response = requests.get(result, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                summary = soup.find("meta", {"name": "description"})
                summary_content = summary["content"] if summary else ""
                # Metni özetle
                summary_text = self.summarize_text(summary_content)
                results_text += f"{summary_text}\n"

        except Exception as e:
            results_text = f"Hata: {e}"

        if results_text:
            with open('Logs/results_logs.txt', 'a') as f:
                f.write(f"{datetime.now()} - {self.query}\n{results_text}\n\n")

            with open('Logs/ml_logs.txt', 'a') as f:
                f.write(f"{datetime.now()} - {self.query}\n")

        self.finished.emit(results_text)

    @staticmethod
    def summarize_text(text):
        # Metni işle
        doc = nlp(text)
        # Cümleleri al ve uzunluklarına göre sırala
        sentences = [sent.text for sent in doc.sents]
        sentences.sort(key=len, reverse=True)
        # İlk cümleyi döndür (en uzun cümle)
        return sentences[0] if sentences else ""