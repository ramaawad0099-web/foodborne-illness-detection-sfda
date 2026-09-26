# AI-Powered Foodborne Illness Detection Platform
*In Collaboration with the Saudi Food and Drug Authority (SFDA)*

## Project Overview
Foodborne illness outbreaks pose a significant challenge to public health systems, requiring rapid identification and proactive intervention. Developed in collaboration with the Saudi Food and Drug Authority (SFDA), this platform provides an automated, data-driven framework for real-time outbreak monitoring, predictive risk modeling, and early warning detection. The system transforms raw health incident data and social media signals into actionable insights to support evidence-based public health decisions.

---

## Technical Highlights and Methodology
- **Multi-Source Data Ingestion Pipeline:** Automated scrapers and API integrations (via Apify for X/Twitter and OpenFDA) to ingest real-time public sentiment, health reports, and regulatory recall data.
- **Advanced NLP & Transformer Models:** Customized State-of-the-Art Transformer architectures fine-tuned for specialized tasks:
  - **BioBERT** for Biomedical Named Entity Recognition (NER) and BIO tagging.
  - **BERTweet** for real-time social media Anomaly and Alert Detection.
  - **RoBERTa-Large** for automated Product Category Classification.
- **Explainable AI (XAI):** Integrated **LIME** (Local Interpretable Model-agnostic Explanations) to offer model transparency, highlighting critical tokens driving model predictions for regulatory confidence.
- **Interactive Geospatial Dashboard:** Custom Streamlit interface featuring interactive Folium maps for spatial-temporal outbreak tracking, risk severity heatmaps, and automated alert triggers.
- **Robust Model Evaluation:** Rigorous benchmarking using `seqeval` for sequence labeling tasks, multi-class confusion matrices, and standard classification metrics ($F_1$-Score, Precision, Recall).

---

## Technical Stack
- **Core Language:** Python
- **Deep Learning & NLP:** PyTorch, Hugging Face (`transformers`, `datasets`, `accelerate`), BioBERT, BERTweet, RoBERTa-Large, NLTK, spaCy
- **Explainable AI (XAI):** LIME
- **Data Engineering & Scraping:** Pandas, NumPy, Apify Client (`apify-client`), Requests (openFDA API), Emoji
- **Machine Learning & Evaluation:** Scikit-learn, Seqeval
- **Visualization & Geospatial Analytics:** Streamlit, Folium (`streamlit-folium`), Matplotlib, Seaborn
- **Environment & Deployment:** Python-dotenv, Secrets Management

---

## Author
**Rama Alshammari**  
*Data Scientist | AI & Machine Learning Engineer*  
- **Email:** ramaawad0099@gmail.com  
- **GitHub:** https://github.com/ramaawad0099-web
## Exploratory Data Analysis (EDA)

### Raw Tweet Length Distribution (Pre-Preprocessing)
![Tweet Length Distribution](1.png)

> **Note on Data State:** This exploratory analysis was performed on the **raw, uncleaned dataset** prior to text preprocessing and tokenization. 

- **Dataset Scope:** Analyzed 4,122 raw tweets directly after multi-source ingestion[cite: 9].
- **Key Observation:** The character distribution peaks around typical Twitter limits with a mean length of 154 characters[cite: 9].
- **Engineering Value:** Identifying this baseline distribution guided the subsequent text-cleaning steps and established the optimal `max_length` parameter for the **BERTweet** model, preventing context truncation while maintaining processing efficiency[cite: 9].
