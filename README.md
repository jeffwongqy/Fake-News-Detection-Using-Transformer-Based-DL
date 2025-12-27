# Fake News Detection Using Transformer-Based Deep Learning Models on the ISOT Dataset

# Aim
This project aims to design, develop, and deploy an efficient fake news classification system using transformer-based deep learning algorithms trained on the ISOT dataset, enabling accurate detection of fake and real news articles through an interactive Gradio-based web application. 

# Objectives
1. To study the problem of fake news dissemination and understand its impact on information credibility in digital media.
2. To explore and preprocess the ISOT Fake News Dataset, including data cleaning, text normalization, and label preparation for model training.
3. To implement transformer-based deep learning models for effective feature extraction and text classification.
4. To train and evaluate the transformer models using appropriate performance metrics such as accuracy, precision, recall, and F1-score.
5. To develop an interactive user interface using the Gradio application, ensuring usability, responsiveness, and reliability.
6. To demonstrate the practical applicability of transformer-based deep learning techniques in combating misinformation.

# Preprocessing 
1. Text Cleaning for TRUE & FAKE dataset
- lowercasing
- headline-content separation (only for TRUE dataset)
- whitespace normalization
- digital removal
- removal of punctuation and special characters

2. Class Label Assignment
Class labels are added to both datasets: articles from the TRUE dataset are labeled as "TRUE," and articles from the FAKE dataset are labeled as "FAKE".

3. Dataset Combination
The TRUE and FAKE datasets are combined into a single dataset using concatenation. This unified dataset contains cleaned text samples along with their corresponding class labels.

4. Tokenization

To convert textual data into a numerical form suitable for deep learning models.
1. A tokenizer is created with a vocabulary size of 20,000 words.
2. An Out-of-Vocabulary (OOV) token is used to handle unseen words.
3. The tokenizer is fitted on the cleaned text.
4. Each news article is converted into a sequence of integers representing word indices.

5. Sequence Padding

Since text sequences vary in length, padding is applied to ensure uniform input size:
- A maximum sequence length of 256 tokens is defined.
- Sequences shorter than the maximum length are padded with zeros at the end.
- Longer sequences are truncated to fit the defined length.

6. Label Encoding
Since class labels are categorical, label encoding is applied - to convert both TRUE and FAKE labels into numerical values and the encoder is fitted on training labels and applied to test labels to prevent data leakage. 

