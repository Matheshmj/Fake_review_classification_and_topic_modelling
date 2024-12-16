# Fake_review_classification_and_topic_modelling

## Overview
This project focuses on detecting fake reviews and analyzing the underlying themes in the dataset through topic modeling. By leveraging advanced natural language processing (NLP) techniques and machine learning models, the project aims to classify reviews as genuine or fake while extracting prominent topics that provide deeper insights into the text.

## Key Features
The project incorporates two main functionalities. The first is fake review classification, where reviews are categorized into genuine or fake using traditional machine learning models and fine-tuned transformer models like DistilBERT. The second is topic modeling, which identifies recurring themes within the reviews using methods such as Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF). These features enable a comprehensive understanding of the data and its inherent patterns

## Dataset

The dataset used in this project consists of the following columns:

**Category:** The category of the product or service being reviewed.

**Rating:** The numerical rating provided by the reviewer.

**Label:** Indicates whether the review is genuine or fake.

**Text:** The actual text content of the review.

## Tools and Technologies
The project is implemented using Python, utilizing libraries such as pandas and NumPy for data analysis. NLP tasks are handled using NLTK and spaCy, while scikit-learn is employed for machine learning. For topic modelling, gensim is used, and visualization is performed with matplotlib, seaborn, and wordcloud.

## Machine Learning Models
The following machine learning models were used for fake review classification,

1.Logistic Regression

2.Decision Tree Model

3.Random Forest Classifier

4.Support Vector Machine (SVM)

5.Naive Bayes

## Deep Learning Models
The following deep learning models were implemented for advanced classification tasks:

1.RNN Model

2.LSTM Model

3.Bi-LSTM Model

## Transformers
Transformers were utilized to enhance the performance of fake review classification tasks. The details of the transformer-based approach are as follows

### Data Preparation

**Data Preprocessing**: Text data is cleaned, tokenized, and prepared for input into the transformer model.

**Tokenization:** DistilBERT tokenizer is used to split text into tokens suitable for model input.

**Truncation and Padding:** Ensures all input sequences are of the same length for efficient processing.

### Model Selection

**Pre-trained Model:** The project utilizes DistilBERT (distilbert-base-uncased), a lighter and faster version of BERT, to maintain performance with reduced computational cost.

## Topic Modelling
Topic modelling techniques were used to uncover hidden themes within the reviews. The details are as follows:

### Data Preparation

Reviews were cleaned and preprocessed to extract meaningful information for topic analysis.

### Feature Extraction

Key features were extracted from the text data for use in the topic modelling algorithms.

### Model Training

Two main models were implemented:

**1.Latent Dirichlet Allocation (LDA):** Used to identify topics by grouping words with similar meanings.

**2.Non-Negative Matrix Factorization (NMF):** Used to extract meaningful components from the review data.

### Topic Display and Visualization

**WordCloud:** Created for each topic to visualize the most frequent words.

## Results
The classification model achieved 90% accuracy in detecting fake reviews. Key topics identified include product quality, delivery experience, and customer service. Visual insights like word clouds and bar charts enhance interpretability of the findings.

## Future Improvements
Future enhancements could include integrating deep learning models like LSTMs or BERT for better classification performance. An end-to-end pipeline for real-time fake review detection can be developed. Additional exploration of topic modelling techniques, such as Non-Negative Matrix Factorization (NMF), can also be undertaken

## contact 
*mail id:* mathesh312020@gmail.com
*linkedin id:* linkedin.com/in/mathesh-m-75b752202/ 
