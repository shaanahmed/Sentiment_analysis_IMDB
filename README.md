dataset: https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews

This project focuses on Sentiment Analysis on the IMDB Dataset, aiming to analyze and classify user reviews as either positive or negative using machine learning and deep learning techniques. The goal is to explore the application of Natural Language Processing (NLP) in understanding sentiments expressed in text data, providing a robust solution for automating sentiment classification.

The project employs Python as the primary programming language and utilizes various libraries such as Pandas, Numpy, and Scikit-learn for data preprocessing and machine learning, along with TensorFlow/Keras for building a deep neural network. The IMDB dataset, a comprehensive collection of movie reviews, serves as the foundation for training and testing models. Key steps include data cleaning, text preprocessing (like tokenization and vectorization), and splitting the data into training and testing sets.
A variety of models have been implemented, including K-Nearest Neighbors (KNN), K-Means Clustering, Decision Tree, Random Forest, and a Neural Network (Multi-Layer Perceptron). Each model has been evaluated based on metrics such as accuracy, precision, recall, and F1 score, with the goal of achieving over 85% accuracy. The neural network architecture leverages the Softmax activation function to enhance multi-class classification tasks, demonstrating the capability of deep learning in sentiment analysis.

This project showcases the integration of NLP with machine learning and deep learning techniques, emphasizing the importance of preprocessing and model evaluation in achieving accurate results. It also highlights the practical application of these techniques in understanding and analyzing unstructured text data. Future enhancements could include expanding the dataset, incorporating word embeddings like GloVe or Word2Vec, and implementing real-time sentiment analysis for broader applications in social media monitoring, customer feedback systems, and recommendation engines.

SENTIMENT ANALYSIS ON IMDB DATASET

The primary goal of this project is to perform sentiment analysis on IMDB movie reviews using multiple machine learning and deep learning models. The project evaluates the performance of various models to determine the most effective one for this task.

Dataset (https://shorturl.at/QwLEq)

Source: IMDB dataset with 50,000 movie reviews.
Structure:
Columns: review (textual review) and sentiment (labels: positive/negative).
Total rows: 50,000.
No missing values observed.


Steps and Methodologies

1. Exploratory Data Analysis (EDA)
Dataset Overview:
Unique Reviews: 49,582.
Class Distribution: 25,000 positive and 25,000 negative reviews (balanced dataset).
Visualization: Class distribution visualized using Seaborn's countplot.

2. Data Cleaning
No missing values detected; no additional cleaning performed.

3. Text Preprocessing
Label Encoding: Sentiment labels (positive, negative) encoded to numerical values.
Text Vectorization:
Used TF-IDF Vectorizer with a maximum feature limit of 5,000.
Converts text reviews into a numerical format for machine learning models.

4. Data Splitting
Train-Test Split: 80% training and 20% testing.


Models Implemented

1. K-Nearest Neighbors (KNN)
Algorithm: Classification based on the majority vote among nearest neighbors.
Accuracy: 73.08%.
Confusion Matrix: Visualized using Matplotlib.

2. K-Means (Unsupervised Clustering)
Algorithm: Clustering based on Euclidean distance.
Accuracy: 48.38% (low due to unsupervised nature).

3. Decision Tree Classifier
Algorithm: Tree-based classification.
Accuracy: 71.41%.
Confusion Matrix: Visualized.

4. Random Forest Classifier
Algorithm: Ensemble of Decision Trees.
Accuracy: 85.18%.
Confusion Matrix: Visualized.

5. Neural Network (Multi-Layer Perceptron)
Architecture:
Input Layer: 5,000 features.
Two hidden layers: 128 and 64 neurons, ReLU activation.
Output Layer: Softmax activation for binary classification.
Optimizer: Adam.
Loss Function: Sparse Categorical Crossentropy.
Performance:
Final Accuracy: 87.85%.
Validation Loss: Indicates some overfitting.

Model Evaluation

Metrics: Accuracy, Precision, Recall, F1 Score.
Results Summary:
Best Model: Neural Network with the highest accuracy (87.85%).
Random Forest also performed well with an accuracy of 85.18%.
K-Means, as expected for an unsupervised model, showed the lowest performance.
Comparative Analysis
Performance comparisons across all models were visualized using bar plots for each metric (Accuracy, Precision, Recall, F1 Score).



Future Scope

Incorporating advanced Natural Language Processing (NLP) techniques such as word embeddings (e.g., Word2Vec, GloVe).
Experimenting with deep learning architectures like RNNs or Transformers for potentially better results.
Hyperparameter tuning to further improve model performance and reduce overfitting.

The project successfully implemented and compared various machine learning and deep learning models for sentiment analysis.

The Neural Network outperformed other models but demonstrated some overfitting, evident from the increasing validation loss.

Random Forest emerged as a robust alternative with high accuracy and balanced performance across all metrics.




