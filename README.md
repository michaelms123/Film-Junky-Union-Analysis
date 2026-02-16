# Film Junky Union — IMDb Review Sentiment Classifier
## NOTE: ## 
I could not install Intels GPU extension for Torch and could not run the BERT portion of my project locally. I used Colab to access a GPU compatible with torch 'cuda'. This notebook in my repo is a copy of the Colab Notebook. 
**You can find the Colab Notebook here: https://colab.research.google.com/drive/1ieHZcyKJB5AFtj6L6L5kYCrV_rRTTDTm?usp=sharing** 

## Project Overview: ##
The Film Junky Union, a community for classic movie enthusiasts, is developing a system to filter and categorize movie reviews.
The goal of this project is to train a machine learning model that can automatically detect negative IMDb movie reviews.

This is a binary text classification task using labeled IMDb reviews. The primary evaluation metric is F1-score, with a required threshold of: **F1 >= 0.85**

## Objectives: ##
This project follows an end-to-end supervised ML workflow:
- Load and inspect the dataset
- Preprocess text data for modeling
- Perform exploratory data analysis (EDA)
- Evaluate class balance / imbalance
- Train and compare at least 3 different models
- Evaluate performance on a held-out test set
- Write custom reviews and classify them using each model
- Analyze differences between model results
- Present conclusions and final recommendations

## Dataset:
- The dataset is stored in:
  **imdb_reviews.tsv**

## Target Variable:
- **pos**
  - 0 = negative review
 - 1 = positive review

## Dataset Split Column:
- ds_part
- train / test

## Feature:
- review (raw review text)

## Tools & Libraries:
- This project uses Python and standard NLP / ML libraries:
  - pandas, numpy
  - matplotlib, seaborn
  - scikit-learn
  - nltk / spacy 
  - LightGBM
  - BERT embeddings (GPU from Colab used)
 
## Workflow:
1) Data Loading
- Loaded .tsv dataset
- Checked shape, missing values, and label distribution

2) Exploratory Data Analysis (EDA)
- Review length distributions
- Target class balance analysis
- Visual exploration of dataset patterns

3) Text Preprocessing
- Lowercasing
- Removing punctuation
- Removing stopwords
- Lemmatization

4) Vectorization
- Text was transformed into numeric features using:
  - TF-IDF
  - BERT embeddings 

5) Modeling
- Logistic Regression (Baseline)
- RandomForestClassifier
- LightGBM (gradientboosting)
- BERT embeddings in conjunction with classifier

6) Evaluation
- Models were evaluated using:
  - F1 Score (primary metric)
  - PRC
  - ROC-AUC (optional)
 
7) Custom Review Testing
A set of original movie reviews were written and tested against each model to compare:
- Model consistency
- Sensitivity to sarcasm, tone, and wording
- Differences between validation and real-world examples

## Results
- The project successfully trained sentiment classification models meeting the performance requirement:
  Target achieved: F1 ≥ 0.85
- Model performance was compared and discussed in the notebook, including explanations for why certain models performed better or worse.
