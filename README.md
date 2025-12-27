# Fake News Detection Using Transformer-Based Deep Learning Models on the ISOT Dataset

# Aim:
This project aims to design, develop, and deploy an efficient fake news classification system using transformer-based deep learning algorithms trained on the ISOT dataset, enabling accurate detection of fake and real news articles through an interactive Gradio-based web application. 

# Objectives:
1. To study the problem of fake news dissemination and understand its impact on information credibility in digital media.
2. To explore and preprocess the ISOT Fake News Dataset, including data cleaning, text normalization, and label preparation for model training.
3. To implement transformer-based deep learning models for effective feature extraction and text classification.
4. To train and evaluate the transformer models using appropriate performance metrics such as accuracy, precision, recall, and F1-score.
5. To develop an interactive user interface using the Gradio application, ensuring usability, responsiveness, and reliability.
6. To demonstrate the practical applicability of transformer-based deep learning techniques in combating misinformation.

# Dataset:
The datasets used in this project were extracted from the ISOT Fake News Detection repository, available at the Online Academic Community of the University of Victoria (https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)

# Preprocessing:
1. Text Cleaning for TRUE & FAKE dataset
- lowercasing
- headline-content separation (only for TRUE dataset)
- whitespace normalization
- digital removal
- removal of punctuation and special characters

2. Class Label Assignment: 
- Class labels are added to both datasets: articles from the TRUE dataset are labeled as "TRUE," and articles from the FAKE dataset are labeled as "FAKE".

3. Dataset Combination: 
- The TRUE and FAKE datasets are combined into a single dataset using concatenation.
- This unified dataset contains cleaned text samples along with their corresponding class labels.

4. Tokenization:
- To convert textual data into a numerical form suitable for deep learning models.
- A tokenizer is created with a vocabulary size of 20,000 words.
- An Out-of-Vocabulary (OOV) token is used to handle unseen words.
- The tokenizer is fitted on the cleaned text.
- Each news article is converted into a sequence of integers representing word indices.

5. Sequence Padding:
- Since text sequences vary in length, padding is applied to ensure uniform input size:
- A maximum sequence length of 256 tokens is defined.
- Sequences shorter than the maximum length are padded with zeros at the end.
- Longer sequences are truncated to fit the defined length.

6. Train-Test Split:
- The dataset is divided into training and testing sets: 80% of the data is used for training, and 20% is reserved for testing.
- A fixed random state ensures reproducibility. 

9. Label Encoding:
- Since class labels are categorical, label encoding is applied to convert both TRUE and FALSE labels into numerical values.
- The encoder is then fitted on the training labels and applied to the test labels to prevent data leakage.

11. Saving the Preprocessed Data:
- The training and testing datasets are saved as a serialized file.
- The trained tokenizer is saved separately to ensure consistent preprocessing during model inference and Gradio deployment.

# Model Architecture:
A hybrid deep learning model combining Bidirectional LSTM, Transformer block, and Convolutional Neural Network (CNN) layers is designed to effectively capture both sequential and contextual information from news text. 

- The input layer accepts a padded token sequence of fixed length (256 tokens).
- An embedding layer converts token indices into dense vector representations of dimension 64. 
- A Bidirectional LSTM layer captures long-range dependencies from both forward and backward directions in the text.
- A custom Transformer block is implemented using multi-head self-attention, feed-forward networks, residual connections, layer normalization, and dropout for regularization.
- A 1D-Convolutional layer is applied to extract local n-gram level features.
- Global Average Pooling reduces dimensionality and aggregates features.
- Fully connected dense layers with dropout are used for classification.
- A sigmoid output layer produces the final probability of the binary classification (FAKE or TRUE).

This architecture leverages the strengths of recurrent neural networks, attention mechanisms, and convolutional layers to improve fake news detection performance. 

The model is compiled using:
- Adam optimizer with a low learning rate of 1e-5 to ensure stable convergence.
- Binary Cross-Entropy loss function, suitable for a binary classification problem.
- Accuracy as the primary evaluation metric.

ADD ARCHITECTURE IMAGE

# Stratified K-Fold Cross-Validation 
To ensure robustness and reduce bias due to class imbalance, 5-fold stratified k-fold cross-validation is employed:
- The training data is split into 5 folds while preserving class distribution.
- In each fold, four folds are used for training, and one fold is used for validation.
- The model is trained for 10 epochs with a batch size of 256 in each fold.
- Validation accuracy is recorded after each fold.

At the end of cross-validation, the mean validation accuracy is computed as the average of all fold accuracies, providing a reliable estimate of model performance. 

The mean validation accuracy was computed to be 0.97. 

# Validation Split Approach







# Tools: 
- Jupyter Notebook
- Gradio

# References: 


