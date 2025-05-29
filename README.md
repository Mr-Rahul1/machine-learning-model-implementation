# machine-learning-model-implementation

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: RAHUL KAPSE

*INTERN ID*: CT04DK813

*DOMAIN*: PYTHON PROGRAMMING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

# In this task

The goal of this task was to build a machine learning model that can automatically classify SMS messages as spam or not spam (ham). This is a common real-world problem in natural language processing (NLP) and text classification, where the system learns from labeled examples and predicts whether a new message is spam.

1. Data Acquisition
I used a publicly available dataset called the SMS Spam Collection Dataset, which contains 5,574 SMS messages labeled as either “spam” or “ham.” Each message is a short text and has a corresponding label indicating whether it is spam or not. This dataset is widely used for experiments in spam detection because it is clean and well-labeled.

2. Data Preprocessing
The raw SMS text messages need to be converted into a format suitable for a machine learning model:

Label Encoding: The original labels were textual (“ham” and “spam”). I converted these to numeric values (0 for ham and 1 for spam) to simplify the classification process.

Text Vectorization: Machine learning models cannot work directly with raw text, so I transformed the text into numerical features. This was done using CountVectorizer from Scikit-learn, which converts the collection of text messages into a matrix of token counts (bag-of-words representation). Each unique word in the dataset becomes a feature, and the value is the count of how many times that word appears in each message.

3. Train-Test Split
To evaluate the model's performance, the dataset was split into two parts:

Training set (80%): Used to train the model.

Test set (20%): Used to evaluate how well the model performs on unseen data.

This separation is important to avoid overfitting and to simulate real-world performance.

4. Model Training
I chose the Multinomial Naive Bayes classifier because it works well with text classification problems and bag-of-words features. Naive Bayes is based on Bayes’ theorem and assumes independence between features (words), which, despite being a strong assumption, works surprisingly well in practice.

The model was trained on the vectorized training data (X_train_vec) and corresponding labels (y_train).

5. Model Evaluation
Once trained, the model’s predictions were made on the test set (X_test_vec), and several evaluation metrics were calculated:

Accuracy: The percentage of correctly classified messages.

Confusion Matrix: Shows counts of true positives, true negatives, false positives, and false negatives to understand errors.

Classification Report: Provides precision, recall, and F1-score for each class:

Precision indicates how many predicted spam messages were actually spam.

Recall shows how many actual spam messages were correctly detected.

F1-score balances precision and recall.

The model showed high accuracy (about 98%), with very good precision and recall, meaning it reliably detects spam without misclassifying too many legitimate messages.

6. Testing Custom Messages
Finally, the trained model was tested on a few new example messages to show how it classifies unseen input, demonstrating practical use.

# OUTPUT
