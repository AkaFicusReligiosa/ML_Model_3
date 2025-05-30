

# Spam Mail Classifier using Logistic Regression

This project implements a **Logistic Regression-based classification system** to detect whether an email message is **spam or not**. The model is trained on a dataset of labeled emails and uses **text preprocessing and TF-IDF vectorization** to convert raw text into meaningful numerical features.

---

## 🎯 Objective

To develop a machine learning model that accurately classifies incoming emails as **spam** or **ham (not spam)** using logistic regression and natural language processing techniques.

---

## 📁 Dataset

- **Source**: [Kaggle - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Instances**: 5,574 messages labeled as:
  - `spam`: Unwanted, promotional, or fraudulent content
  - `ham`: Legitimate/non-spam messages
- **Format**: `label`, `message`

---

## 🚀 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK (for NLP preprocessing)
- Jupyter Notebook

---

## 🧠 Model Workflow

1. **Load and Inspect Data**
2. **Text Preprocessing**
   - Lowercasing
   - Removing punctuation and stopwords
   - Tokenization
   - Lemmatization (optional)
3. **Vectorization**
   - TF-IDF vectorizer to convert text into feature vectors
4. **Model Training**
   - Logistic Regression using Scikit-learn
5. **Evaluation**
   - Accuracy, Precision, Recall, F1-score, Confusion Matrix
6. **Prediction**
   - Input email and classify as spam or ham

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

---

## 🧪 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spam-mail-classifier.git
   cd spam-mail-classifier
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

   ```bash
   jupyter notebook spam_classifier.ipynb
   ```

---

## 💡 Sample Prediction Code

```python
sample_email = ["Congratulations! You've won a $1000 gift card. Click to claim."]
vector = tfidf.transform(sample_email)
prediction = model.predict(vector)
print("Prediction:", "Spam" if prediction[0] == 1 else "Ham")
```

---

## 📈 Results

* Achieved accuracy of **\~95%** with well-tuned parameters and cleaned data.
* Logistic Regression showed robust performance for text-based binary classification.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

* [Kaggle - SMS Spam Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
* Scikit-learn
* NLTK
* Python open-source community

```
