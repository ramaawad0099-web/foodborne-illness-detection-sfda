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

  ## ## Exploratory Data Analysis (EDA) — Raw Data Insights (TWEET-FID Data)

### Tweet Length Distribution 
![Tweet Length Distribution](1.png)

> **Data Quality & Pre-Preprocessing Note:** This analysis reflects the **raw dataset prior to text cleaning and tokenization**. The TWEET-FID dataset is of exceptional quality, having been annotated by multiple crowdsource workers and rigorously cross-checked by food safety experts to ensure high label accuracy.

- **Dataset Scope:** Analyzed 4,122 expert-verified raw tweets directly post-ingestion.
- **Key Observation:** Exhibits a natural tweet character distribution with a mean length of 154 characters and a peak near standard Twitter limits.
- **Engineering Purpose:** Understanding this raw baseline directly guided our text-cleaning pipeline and informed the selection of the optimal `max_length` parameter for fine-tuning **BERTweet**, preventing context loss while preserving GPU efficiency.



### Class Label Distribution & Balance Verification
![Class Label Distribution](2.png)

- **Class Breakdown:** 2,076 **Alert** tweets (Label 1) vs. 2,046 **Noise** tweets (Label 0).
- **Dataset Balance:** Highly balanced distribution (~50/50 split across binary categories).
- **Engineering Purpose:** Confirms dataset equilibrium, ensuring the classification model trains without class bias or requiring artificial resampling/reweighting techniques.

- ### Named Entity Co-occurrence Matrix
![Entity Co-occurrence Heatmap](3.png)

- **Semantic Richness:** Evaluates joint entity occurrences (`food`, `loc`, `symptom`, `other`) within individual tweets for Named Entity Recognition (NER) pipeline design.
- **Key Observation:** High co-occurrence values between `symptom` and `food` (342 co-occurrences) as well as `symptom` and `loc` (258 co-occurrences).
- **Domain Relevance:** Validates that "Alert" signals represent granular, context-rich public health reports linking specific symptoms to distinct food items and locations, rather than generic slang.
