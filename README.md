# 🎓 AI Kursuse Nõustaja

Tartu Ülikooli kursuste soovitamise RAG-süsteem, mis on ehitatud Streamlit'i ja OpenRouter API peale.

---

## 📌 Projekti kirjeldus

AI Kursuse Nõustaja on vestlusliides, mis aitab tudengitel leida Tartu Ülikooli ainekavast sobivaid kursusi loomuliku keele abil. Kasutaja kirjeldab, mida ta soovib õppida, ning süsteem otsib semantilise otsingu abil kõige sobivamad kursused ja esitab need LLM-i abil struktureeritud soovitusena.

Rakendus kasutab **RAG** (*Retrieval-Augmented Generation*) arhitektuuri:
1. Kasutaja päring muudetakse vektoriks (`BAAI/bge-m3` mudel)
2. Filtritele vastavad kursused järjestatakse kosinussarnasuse järgi
3. Top-5 kursust saadetakse kontekstina LLM-ile (`google/gemma-3-27b-it` OpenRouteri kaudu)
4. LLM genereerib soovituse ainult antud konteksti põhjal

---

## 🚀 Paigaldamine ja käivitamine

### 1. Klooni repositoorium

```bash
git clone https://github.com/markusmuts/tehisintellekti-rakendamise-projekt.git
cd tehisintellekti-rakendamise-projekt
```

### 2. Loo conda keskkond

```bash
conda env create -f environment.yml
conda activate oisi_projekt
```

### 3. Seadista API võti

Loo projekti juurkausta fail `.env` ja lisa sinna oma [OpenRouter](https://openrouter.ai/) API võti:

```
API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxx
```

### 4. Käivita rakendus

```bash
streamlit run ois-projekt.py
```

---


## 🛠️ Kasutatavad tehnoloogiad

| Tehnoloogia | Kasutus |
|---|---|
| [Streamlit](https://streamlit.io/) | Veebirakenduse liides |
| [OpenRouter](https://openrouter.ai/) | LLM API (`google/gemma-3-27b-it`) |
| [sentence-transformers](https://www.sbert.net/) | Tekstivektorid (`BAAI/bge-m3`) |
| [scikit-learn](https://scikit-learn.org/) | Kosinussarnasuse arvutamine |
| [pandas](https://pandas.pydata.org/) | Andmete töötlemine |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | API võtme haldus |
