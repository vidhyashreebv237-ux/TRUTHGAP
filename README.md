# TRUTHGAP: AI Expectation Gap Auditor

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini%202.5%20Flash-orange.svg)](https://aistudio.google.com/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Charts-Plotly-purple.svg)](https://plotly.com/)

A prompt-engineered AI system designed to detect, quantify, and visualize expectation gaps between product advertising claims and ground-truth consumer experiences.

---

## 🌟 Key Features

1. **Automated Claim Extraction**: Employs Role-Based & Few-Shot prompting to strip marketing fluff and extract verifiable product specifications.
2. **Review Ground-Truth Synthesis**: Processes raw text or CSV datasets to extract genuine user sentiment and observed performance per feature.
3. **0–100 Expectation Gap Scoring**: Evaluates claims vs. user reality through Zero-Shot Comparative Reasoning.
4. **Discrepancy Categorization**: Categorizes features into **Low Gap (0–30)**, **Moderate Gap (31–60)**, and **High Gap (61–100)**.
5. **Interactive Plotly Visualizations**:
   - Speedometer Gauge (Overall Gap Score)
   - Feature Discrepancy Bar Chart
   - Multi-Dimensional Radar Profile
   - Claim Veracity Donut Chart (Verified / Exaggerated / Misleading)
6. **Executive Audit Report**: Automated diagnostic report generation in Markdown with instant `.md` download.
7. **Prompt Engineering Traceability**: Real-time inspection of intermediate JSON representations from each prompt stage.

---

## 📁 Project Structure

```text
TRUTHGAP/
├── .env.example              # Template showing where your Gemini API key goes
├── .gitignore                # Protects sensitive API keys and virtual environments
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and setup instructions
├── VIVA_GUIDE.md             # Complete academic defense and viva preparation guide
│
├── data/
│   └── sample_reviews.csv    # Ready-to-use sample dataset for quick testing and demonstrations
│
├── src/
│   ├── __init__.py           # Python package marker
│   ├── config.py             # Configuration and environment variable loader
│   ├── prompts.py            # Central repository of prompt engineering templates
│   ├── gemini_client.py      # Gemini API client with JSON extraction and error handling
│   ├── analyzers.py          # Core analytical pipeline and scoring engine
│   └── visualizer.py         # Plotly visualization components
│
└── app.py                    # Streamlit web dashboard
```

---

## 🚀 Quick Start Guide

### 1. Clone or Open the Workspace
```bash
cd TRUTHGAP
```

### 2. Set Up Virtual Environment & Dependencies
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Configure Gemini API Key
1. Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### 4. Run the Web Dashboard
```powershell
.\venv\Scripts\streamlit.exe run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🧪 Demonstration Mode

In the dashboard sidebar, click **"📥 Load Sample Ad & Reviews"** and then press **"🚀 Run TRUTHGAP Audit"** to run a complete audit on pre-configured earbuds marketing claims vs. 8 authentic customer reviews.

---

## 🎓 Academic Defense & Viva Preparation
Review the complete [VIVA_GUIDE.md](file:///c:/Users/vidhy/OneDrive/Desktop/TRUTHGAP/VIVA_GUIDE.md) for detailed explanations of prompt engineering methodologies, mathematical formulations, and answers to the top 10 viva questions.
