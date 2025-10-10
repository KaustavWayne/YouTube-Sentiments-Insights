# 🎬 YouTube Chrome Comment Analyzer

<p align="center">
  <img src="https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube">
</p>

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3.3-orange?logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-latest-blue?logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow?logo=huggingface&logoColor=black)
![MLflow](https://img.shields.io/badge/MLflow-2.17.0-lightgrey?logo=mlflow&logoColor=black)
![Chrome Extension](https://img.shields.io/badge/Chrome_Extension-yes-red?logo=google-chrome&logoColor=white)

---

## ✨ Overview
The **YouTube Chrome Comment Analyzer** is a Chrome extension and web tool that fetches YouTube comments for any video, performs sentiment analysis, and provides a detailed visualization including:

- Sentiment trend over time 📈
- Word cloud of comments ☁️
- Top comments with sentiment labels 🏆
- Summary metrics like total comments, unique commenters, and average sentiment score

---

## 🚀 Demo
Check out a quick demo of the plugin in action:  

🎥 **[Watch Demo](https://huggingface.co/spaces/dystopiareloaded7/youtube-chrome-plugin)**  

> Tip: You can also embed a short GIF/video of usage here using markdown:
>
> ```markdown
> ![Demo](demo.gif)
> ```

---

## 🛠 Tech Stack
- **Python 3.10**
- **Flask** for backend API
- **Docker** for containerized deployment
- **Hugging Face Spaces** for hosting
- **MLflow** for model tracking
- **Chrome Extension API** for frontend integration
- **YouTube Data API v3** for fetching comments
- **NLTK** for text preprocessing
- **Matplotlib / Wordcloud** for visualizations

---

## ⚡ Features
- Fetch up to 500 YouTube comments per video
- Perform real-time sentiment analysis
- Visualize sentiment distribution, trends, and word cloud
- Display top 25 comments with sentiment scores
- Easy integration with Hugging Face deployment

---

## 💻 Usage
1. Install the Chrome Extension locally or via Hugging Face Space.
2. Open a YouTube video and click the extension icon.
3. View comments analysis and visualizations directly in the popup.

---

## 🗂 Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

---

## 💡 Developer Quotes
> “The best way to predict the future is to analyze the past — one YouTube comment at a time.”  
> — Kaustav Roy Chowdhury

---

## 📄 Author
**Kaustav Roy Chowdhury**  
YouTube Chrome Comment Analyzer © 2025

---

## 📜 License
All rights reserved © 2025 Kaustav Roy Chowdhury
