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

ADD model summary and parameters

# Stratified K-Fold Cross-Validation:
To ensure robustness and reduce bias due to class imbalance, 5-fold stratified k-fold cross-validation is employed:
- The training data is split into 5 folds while preserving class distribution.
- In each fold, four folds are used for training, and one fold is used for validation.
- The model is trained for 10 epochs with a batch size of 256 in each fold.
- Validation accuracy is recorded after each fold.

At the end of cross-validation, the mean validation accuracy is computed as the average of all fold accuracies, providing a reliable estimate of model performance. 

The mean validation accuracy was computed to be 0.97. 

# Validation Split Approach:
A validation split approach is also used for comparison: 
- The same model architecture is trained on the full training dataset.
- 20% of the training data is automatically reserved for validation during training.
- The model is trained for 10 epochs with a batch size of 256.
- Training and validation accuracy are monitored across epochs.

This approach allows faster experimentation and visualization of learning behaviour while maintaining consistency with the cross-validation results. 

# Model Evaluation:
The performance of the proposed transformer-based fake news classification model was evaluated using standard classification metrics, including precision, recall, F1-score, accuracy, and ROC-AUC. 

(A) Classification Report
- The model achieved a high precision of 0.98 for FAKE news, indicating that the majority of news articles predicted as fake were correctly classified.
- For the REAL news class, the model obtained a precision of 0.97 and a recall of 0.98, demonstrating strong capability in correctly identifying real news articles.
- The recall values for both classes (0.97 for FAKE and 0.98 for REAL) show that the model effectively minimizes false negatives.
- The F1-scores of 0.97 (FAKE) and 0.98 (REAL) indicate a balanced trade-off between precision and recall for both classes.
- The model achieved an overall classification accuracy of 98%, correctly classifying 7,730 news articles in the test dataset.

(B) ROC-AUC Curve
- The model achieved a Receiver Operating Characteristic (ROC)–AUC score of 0.996, indicating excellent discrimination capability between fake and real news classes.
- A ROC–AUC value close to 1.0 signifies that the model performs exceptionally well across different classification thresholds.

ADD SCREENSHOT

# Model Deployment:
The trained fake news classification model is deployed using a Gradio web application, providing a simple and interactive interface that allows users to input news text and instantly receive predictions indicating whether the news is FAKE or REAL, thereby demonstrating the model’s practical usability in real-time scenarios.

ADD SCREENSHOT


# Tools: 
- Jupyter Notebook
- Gradio

# References: 
Fake News Detection Datasets | ISOT research lab. (2022). Uvic.ca. https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/
Mo, Y., Qin, H., Dong, Y., Zhu, Z., & Li, Z. (2024). Large Language Model (LLM) AI text generation detection based on transformer deep learning algorithm. ArXiv.org. https://arxiv.org/abs/2405.06652


‌


