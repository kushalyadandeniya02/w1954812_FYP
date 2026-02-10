Context-Aware Sentiment Analysis - Sarcasm and Gen Z Slang Detection in Social Media Text

This repository contains the implementation and resources for the Interim Progress Demonstration (IPD) project titled:
“A Context-Aware Machine Learning Model for Detecting Sarcasm and Gen Z Slangs in Youth-Driven Platforms.”
The project focuses on improving traditional sentiment analysis by incorporating sarcasm awareness and Gen Z slang detection to better interpret youth-driven social media content, particularly from Reddit.

Project Information

Student: Kushalya Dandeniya (w1954812)
Degree: BSc (Hons) Business Data Analytics
University: University of Westminster


Project Aim

The aim of this project is to develop a context-aware sentiment analysis pipeline using Machine Learning and Natural Language Processing techniques that can accurately interpret sarcastic and slang-rich social media comments, enabling more reliable insights into Gen Z engagement and online sentiment.

Key Features

Sarcasm-aware sentiment labelling using VADER polarity inversion
Rule-based Gen Z slang detection using a custom slang dictionary
Context creation by combining parent comments and reply comments
TF-IDF vectorization with integrated metadata features
Exploratory Data Analysis on sarcasm, slang, and sentiment
Prepared pipeline for both traditional ML and transformer-based models

Dataset Overview

Primary Dataset: Sarcasm on Reddit (Kaggle)
Sample Size: 50,000 comments 

Additional Datasets:
genz_slang.csv
all_slangs.csv

The datasets contain sarcastic and non-sarcastic comments with frequent Gen Z slang usage, making them suitable for youth-focused sentiment analysis.

Methodology Summary

Data Cleaning and Preprocessing

Removal of duplicates, missing values, URLs, mentions, hashtags, and short texts
Context creation using parent and reply comments

Slang Detection

Creation of a merged slang dictionary from multiple open-source datasets
Rule-based slang flagging and slang context extraction

Sentiment Labelling

Automated sentiment scoring using VADER
Polarity inversion applied for sarcastic comments to reflect true sentiment

Feature Engineering

TF-IDF vectorization with a maximum of 8,000 features
Metadata feature extraction (text length)
Stratified 70/15/15 train-validation-test split

Exploratory Data Analysis

Distribution analysis and heatmaps for sarcasm, slang, and sentiment

Technologies Used:

Python
Google Colab
Pandas and NumPy
NLTK (VADER Sentiment Analyzer)
Scikit-learn
Matplotlib and Seaborn

Current Project Status

Completed:

Data collection
Data preprocessing and cleaning
Exploratory Data Analysis
Feature engineering and dataset preparation

In progress:

Machine Learning model development
Transformer-based model fine-tuning
Model evaluation and comparison
Dashboard development using PowerBI

Future Work

Train and evaluate traditional ML models (Logistic Regression, SVM, Random Forest)
Fine-tune a transformer model such as DistilBERT
Address class imbalance using SMOTE and class-weighted loss functions
Develop an interactive dashboard for sentiment insights
Final model integration and end-to-end testing
