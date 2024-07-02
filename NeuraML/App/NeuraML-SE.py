import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QTextBrowser,
                             QHBoxLayout, QMessageBox)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QUrl, pyqtSignal, QObject, QThread, QTimer
from googlesearch import search
from bs4 import BeautifulSoup
import requests
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import spacy
from Worker import Worker
import csv

nlp = spacy.load("xx_ent_wiki_sm")
nlp.add_pipe("sentencizer")


class BrAInML(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('NeuraML-SE (2.03.018a)')
        # Sürüm (Bulunduğu kademe|arayüz değişiklikleri|herhangi bir algoritma veya altyapı değişikliği
        # HARF:hangi aşamada olduğu -> a: alpha, b: beta, f: full)
        self.setWindowIcon(QIcon('/Assets/neurazumicon.ico'))
        self.resize(800, 600)
        self.setStyleSheet("background-color: black; color: white;")

        self.query_label = QLabel('<b>Ne Arıyorsunuz?</b>')
        self.query_input = QLineEdit()
        self.query_input.setMaximumHeight(30)
        self.query_input.setStyleSheet("border-radius: 10; background-color: white; color: black")

        self.search_button = QPushButton()
        self.search_button.setIcon(QIcon('/Assets/search.ico'))
        self.search_button.setIconSize(QtCore.QSize(50, 50))
        self.search_button.setStyleSheet("background-color: transparent; border: none; color: white")
        self.search_button.setToolTip('Ara')

        self.evaluate_button = QPushButton('Değerlendir')
        self.evaluate_button.clicked.connect(self.show_evaluation_dialog)
        self.evaluate_button.setStyleSheet("background-color: white; color: black; border-radius: 10; height: 30; font-weight: bold")
        self.evaluate_button.setVisible(False)

        self.result_browser = QTextBrowser()
        self.result_browser.setStyleSheet("border-radius: 10; border: none; background-color: transparent")

        self.result_label = QLabel()
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_label.setFont(QtGui.QFont('Arial', 8, weight=QtGui.QFont.Bold))

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.query_input)
        self.layout.addWidget(self.search_button)

        self.vertical_layout = QVBoxLayout()
        self.vertical_layout.addWidget(self.query_label)
        self.vertical_layout.addLayout(self.layout)
        self.vertical_layout.addWidget(self.result_browser)
        self.vertical_layout.addWidget(self.evaluate_button)
        self.vertical_layout.addWidget(self.result_label)

        self.setLayout(self.vertical_layout)

        self.search_button.clicked.connect(self.search_method)
        self.query_input.returnPressed.connect(self.search_method)
        self.result_browser.anchorClicked.connect(self.open_link)

        # Veri toplama ve model eğitimi
        self.vectorizer = TfidfVectorizer()
        self.model = RandomForestClassifier()

        self.nlp = spacy.load("en_core_web_sm")
        # Modeli eğit
        self.model = self.create_ml_model()
        #Önbellek sözlüğü
        self.cache = {}
        #Sonuçları saklamak için sözlük
        self.result_cache = {}
        self.thread = QThread()
        self.worker = None
        self.results_file = 'Logs/search_results.csv'

    def save_search_result(self, query, result):
        with open(self.results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([query, result])

    def create_ml_model(self):
        X_train, y_train = self.load_data()

        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        model = RandomForestClassifier()
        model.fit(X_train_tfidf, y_train)
        return model

    @staticmethod
    def load_data():
        X_train = []
        y_train = []

        # results_logs.txt'den verileri oku
        with open('Logs/results_logs.txt', 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(' - ')
            if len(parts) == 2:
                date_str, query = parts
                try:
                    # Tarihi parse etmeye çalış
                    date = np.datetime64(date_str)
                    X_train.append(query)
                    y_train.append(date)
                except ValueError:
                    # Hatalı tarih formatı durumunda geç
                    continue

        return X_train, y_train

    def process_feedback(self, feedback):
        if feedback == "evet":
            # Modeli güncelleme işlemi
            # Örnek olarak: self.model.partial_fit(X_new, y_new)
            self.update_feedback("evet")  # Mevcut girdi için çıkan sonuçları modele kaydet
        elif feedback == "hayır":
            # Farklı sitelerden yeni sonuçlar almak için modeli güncelleme işlemi
            # Yeni veri eklemek için:
            # X_new, y_new = self.load_additional_data()
            # self.model.partial_fit(X_new, y_new)
            self.update_feedback("hayır")  # Mevcut girdi için çıkan sonuçları modele kaydetme
            # Sonuçları sıfırla ve yeni sonuçlar için arama yap
            self.result_browser.clear()
            self.search_method()

    def search_method(self):
        query = self.query_input.text()

        if self.worker:
            self.worker.finished.disconnect(self.show_results)
            self.worker.deleteLater()

        self.worker = Worker(query)
        self.worker.finished.connect(self.show_results)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

        X_query = self.vectorizer.transform([query])
        prediction = self.model.predict(X_query)
        self.result_label.setText(f"Tahmin: {prediction} - {query}")

        self.result_browser.setHtml("<h4><i>İşleniyor...</i></h4>")

    def show_results(self, results_text):
        self.thread.quit()
        self.thread.wait()

        if results_text:
            self.result_browser.setHtml(results_text)
            self.evaluate_button.setVisible(True)
            self.save_search_result(self.query_input.text(), results_text)
        else:
            QtWidgets.QMessageBox.warning(self, "Hata", "Model Yanıt Vermedi")

    def show_evaluation_dialog(self):
        feedback, ok = QtWidgets.QInputDialog.getItem(self, "Geri Bildirim", "Sonuçlar yararlı mıydı?",
                                                      ["Evet", "Hayır"], 0, False)
        if ok:
            self.process_feedback(feedback)

    @staticmethod
    def open_link(link):
        QtGui.QDesktopServices.openUrl(QUrl(link))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QFont("Arial", 10)
    app.setFont(font)
    window = BrAInML()
    window.show()
    sys.exit(app.exec_())
