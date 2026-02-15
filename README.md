Here’s a **clear, brief, copy-and-paste–ready version** of your project description 👇

---

# **AI-Powered GitHub Self-Analysis Dashboard**

## 🚀 Overview

An interactive dashboard that analyzes your GitHub profile using **local LLMs (Ollama)** and **data science techniques** to provide insights into your coding habits, skills, and activity trends.

---

## ✨ Key Features

### 🔹 Data Collection

* Fetches repositories, commits, and README files via the GitHub API.

### 🔹 LLM-Based Analysis (Ollama)

* Sentiment analysis of commit messages
* Skill extraction from README files
* Code quality reviews

### 🔹 Traditional Data Science

* Repository clustering (stars, forks, size)
* Commit activity forecasting using Prophet

### 🔹 Dashboard

* Interactive Streamlit UI
* Plotly visualizations
* Local LLM model comparison (Llama 3.1 vs Mistral)

---

## 🛠 Prerequisites

* Python 3.10+
* Ollama installed and running
* Git
* Pull models:

  ```bash
  ollama pull llama3.1
  ollama pull mistral
  ```

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone <repository_url>
cd ai-github-dashboard
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment

* Copy `.env.example` → `.env` (optional)
* Set:

  * `GITHUB_TOKEN`
  * `GITHUB_USERNAME`

---

## ▶️ Usage

### Start Ollama

```bash
ollama serve
```

### Run the Dashboard

```bash
streamlit run app/dashboard.py
```

### Explore

* Enter GitHub username and token
* Click **Fetch Data**
* View insights across dashboard tabs

---

## 📂 Project Structure

```
app/              Streamlit dashboard  
src/              Core logic modules  
  ├─ data_collection.py   GitHub API fetcher  
  ├─ llm_analysis.py       Ollama integration  
  ├─ traditional_ds.py     Clustering & forecasting  
data/             Stored JSON data  
notebooks/         EDA notebooks  
tests/              Unit tests  
```

---

## 📄 License

MIT

---

If you want, I can also make a **short GitHub README version** or a **one-paragraph project summary for your CV/portfolio**.
