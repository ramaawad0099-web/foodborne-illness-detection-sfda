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



## ## Exploratory Data Analysis (EDA) — Raw Data Insights (SemEval-2025 Task 9 Data)

### Data Quality & Missing Value Audit
- **Data Integrity:** 0% missing values detected across all attributes, ensuring consistent baseline records.
- **Deduplication:** Only 18 duplicate rows identified, confirming negligible repetition risk without distorting the overall distribution.
- **Multilingual & Symbol Heterogeneity:** Identified 4,985 texts and 1,950 titles containing non-ASCII symbols and irregular character usage, reflecting the multilingual nature of the dataset.

### Structural Text & Title Length Distribution
![SemEval Text and Title Length Boxplots](4.png)

- **Document Text Length Analysis :**
  - **Median Length:** 1,946 characters (Broad spread with $\text{IQR} = 1560.5$).
  - **Outlier Detection:** 202 unusually long document texts identified as upper outliers; no short-text anomalies or empty fields found.
- **Title Length Analysis :**
  - **Median Length:** 86 characters.
  - **Outlier Detection:** 9 exceptionally short titles and 122 exceptionally long titles detected, with zero missing values.
- **Engineering Value:** Validated structural stability and text boundaries for long-sequence inputs, guiding tokenization truncation thresholds and text-cleaning pipelines.


### Temporal Distribution Analysis
![Temporal Distribution of Reports](5.png)

- **Temporal Trend :** Shows an uneven temporal distribution spanning from 1994 to recent years.
  - **Early Baseline (1994–2005):** Sparse reporting density during early digital surveillance years.
  - **Growth & Surge (Post-2009 & Post-2012):** A sharp increase in volume starting after 2009, with significant activity peaks reaching up to 568 reports in specific peak years (e.g., 2016).
- **Domain Interpretation:** The upward trajectory reflects systemic improvements in digitized reporting infrastructure and heightened regulatory surveillance over time, rather than an organic increase in actual foodborne illness incidents.


### Geographical Report Distribution
![Country Distribution Plot](6.png)

- **Regional Breakdown :** Reporting volume is concentrated across key Anglophone public health jurisdictions:
  - **United States (`us`):** 2,195 reports (Primary contributor)
  - **Australia (`au`):** 921 reports
  - **Canada (`ca`):** 856 reports
  - **United Kingdom (`uk`):** 687 reports
- **Geographic Representation:** Demonstrates regional dominance in public health logging, informing regional risk-normalization strategies when modeling international outbreak trends.


### Product Category Distribution Analysis
![Product Category Distribution](7.png)

- **Category Breadth :** Covers **22 distinct product categories**, demonstrating comprehensive monitoring across food types.
- **Primary Contributor Breakdown:**
  - **Meat, Egg, and Dairy Products:** $28.2\%$ (Largest share)
  - **Cereals and Bakery Products:** $13.2\%$
  - **Fruits and Vegetables:** $10.5\%$
  - **Prepared Dishes and Snacks:** $9.2\%$
  - **Low-Risk Categories:** Items like sugars and syrups each contribute less than $1\%$.
- **Regulatory Alignment:** Reflects highly targeted surveillance focused on high-risk, perishable, and animal-derived food groups susceptible to microbial contamination.


### Granular Product Specificity Analysis
![Top 10 Products Plot](8.png)

- **Product Diversity :** Identifies **1,022 unique product names**, demonstrating extremely high specificity in public health incident logging.
- **Frequent High-Risk Specific Products:**
  - **Ice Cream:** 185 instances
  - **Chicken-based Products:** 138 instances
  - Followed by cakes, ready-to-eat meals, cookies, cheese, salads, ground beef, salmon, and peanuts.
- **Long-Tail Distribution:** While high-risk specific items recur frequently, the vast majority of unique products appear only once, emphasizing the necessity of robust NLP entity-extraction pipelines capable of handling rare and unseen vocabulary.


### Hazard Type & Allergen Risk Analysis
![Top 10 Hazards Plot](9.png)

- **Hazard Diversity (Figure 3.12):** Encompasses **128 unique hazard types**, providing comprehensive coverage across biological, chemical, and physical food safety threats.
- **Dominant Pathogens & Allergenic Hazards:**
  - **Listeria monocytogenes:** $13\%$ of total recorded hazards
  - **Salmonella:** $12.2\%$
  - **Milk and Products Thereof (Allergen):** $11.6\%$
  - **Moderate Frequency Risks:** *Escherichia coli*, peanut allergens, gluten/wheat allergens, plastic fragments, soy, and metal fragments.
- **Long-Tail Pattern:** A heavy concentration in primary biological pathogens and top-tier allergens, with a long tail of minor physical contaminants appearing only once or twice.




---


## ## Exploratory Data Analysis (EDA) — Raw Data Insights (FDA Data)
### Raw openFDA API Record Structure

- **Data Origin & Ingestion :** Highlights raw recall, product, and location records fetched directly via the **openFDA Food Enforcement API** prior to text normalization or schema restructuring.
- **Structural Attributes:** Retains original API JSON/record key-value mappings covering essential regulatory metadata:
  - **Recall-Related:** `recall_number`, `reason_for_recall`, `status`, `classification`.
  - **Product-Related:** `product_description`, `code_info`.
  - **Location & Entity Metadata:** `recalling_firm`, `city`, `state`, `country`.
- **Engineering Value:** Validates the raw API payload schema, informing the automated ingestion pipeline, field selection, and preprocessing parser designed to format heterogeneous government records for downstream analysis.


### Monthly Recall Activity Trend Analysis
![Monthly Distribution of FDA Food Recalls](10.png)

- **Temporal Trend :** Demonstrates the monthly fluctuations and longitudinal reporting volume of FDA food recall events.
- **Pattern Identification:** Reveals noticeable seasonality and irregular reporting spikes across different months rather than an even temporal spread.
- **Domain & Engineering Insight:** Identifying these longitudinal patterns provides crucial context for time-series aggregation, enabling the evaluation of regulatory enforcement cycles and potential seasonal outbreak surges.
- 

### Recall Case Status Distribution
![Before Preprocessing: Recall Status Distribution](11.png)

- **Status Breakdown :** Illustrates the operational state of extracted FDA recall cases prior to data preprocessing:
  - **Terminated:** 577 records (Majority of recall cases officially resolved and closed).
  - **Ongoing:** 377 records (Active recall actions undergoing regulatory processing).
  - **Completed:** 46 records (Recalls where all actions are finished but final administrative termination is pending).
- **Engineering Value:** Categorizes the active context of enforcement data, enabling label filtering and status-based feature engineering during data modeling.


### Text Length Variability Analysis (Pre-Preprocessing)
![Before Preprocessing: Raw Text Length Distribution](12.png)

- **Text Length Comparison :** Evaluates length distributions for core textual fields (`product_description` vs. `reason_for_recall`):
  - **`product_description`:** Exhibits higher variability and longer lengths, with extreme right-skewed outliers (up to ~4,000 characters) due to detailed batch codes and embedded product specifications.
  - **`reason_for_recall`:** Displays a significantly narrower text-length range, remaining concise and concentrated near lower character bounds.
- **Engineering Value:** Evaluates sequence length variability before text concatenation, establishing context-length baselines to prevent truncation of critical entity identifiers in the classification transformer model.


### Missing Value Audit (Pre-Preprocessing)
![Before Preprocessing: Missing Values by Column](14.png)

- **Completeness Audit :** Identifies missing value frequencies across all raw API attributes before preprocessing:
  - **Zero Missing Core Attributes:** Essential core fields required for model inputs (`product_description`, `reason_for_recall`, `recall_number`, `classification`, `status`, and `report_date`) contain **$0$ missing values**.
  - **Optional Field Missingness:** Missing data is isolated to optional administrative attributes: `termination_date` ($423$ missing), `more_code_info` ($323$ missing), and `center_classification_date` ($1$ missing).
- **Engineering Value & Preprocessing Rule:** Confirms that data filtering can strictly target records missing mandatory classification text, safely preserving records with missing optional fields without risking training sample size loss.
