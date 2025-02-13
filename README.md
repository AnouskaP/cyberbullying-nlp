# Cyberbullying Detection using NLP and Deep Learning

## Project Overview
This project focuses on **detecting and classifying cyberbullying types** using **Natural Language Processing (NLP)** and **Deep Learning models**, including **Naive Bayes, Logistic Regression, and BERT**. The dataset consists of tweets labeled with different cyberbullying categories.

This project is part of the research paper **"Cyber-bullying Types Detection on Twitter: A Comparative Study of Machine Learning Models"**, which explores the effectiveness of various machine learning models in cyberbullying detection.

## Dataset
- The dataset contains tweets categorized into the following cyberbullying types:
  - **Not Cyberbullying**
  - **Ethnicity-Based Cyberbullying**
  - **Gender-Based Cyberbullying**
  - **Age-Based Cyberbullying**
  - **Religion-Based Cyberbullying**
- The dataset undergoes preprocessing, feature extraction, and model training.

## Project Workflow
### 1. **Data Preprocessing**
   - Removing duplicates and null values.
   - Cleaning text by:
     - Removing special characters, URLs, and numbers.
     - Tokenization, Lemmatization, and Stopword removal using **spaCy**.
   - Generating **TF-IDF features** for text representation.

### 2. **Exploratory Data Analysis (EDA)**
   - **Distribution of cyberbullying types** (bar plots & pie charts).
   - **Word Clouds** to visualize prominent words in tweets.
   - **TF-IDF vectorization** for feature representation.

### 3. **Machine Learning Models**
   - **Naive Bayes Model:**
     - Achieved **high accuracy** in initial classification.
     - Evaluated using **confusion matrix and classification report**.
   - **Logistic Regression Model:**
     - Used **TF-IDF vectorization**.
     - Improved classification performance compared to Naive Bayes.

### 4. **BERT-based Deep Learning Model**
   - Used **BERT (bert-base-cased) tokenizer** for feature extraction.
   - Built a **deep learning model** using **TensorFlow & Keras**.
   - **Fine-tuned BERT** for cyberbullying classification.
   - Achieved **high classification accuracy** after training.

### 5. **Performance Evaluation**
   - **Confusion Matrices** for model performance visualization.
   - **Classification Reports** with **Precision, Recall, and F1-score**.
   - Comparison of ML and BERT models for best accuracy.

## Installation & Dependencies
To run this project, install the required Python libraries:

```bash
pip install tensorflow==2.8.0rc0 transformers==4.20.1 numpy pandas seaborn matplotlib nltk sklearn imblearn wordcloud spacy
```

Download the necessary NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage
Clone the repository and run the Jupyter Notebook:

```bash
git clone https://github.com/your-username/cyberbullying-nlp.git
cd cyberbullying-nlp
jupyter notebook
```

Then, open **"cyberbullying_types_detection(final).ipynb"** in Jupyter Notebook and execute the cells sequentially.

## Results & Findings
- **Naive Bayes Model:** Performed well but had some misclassifications.
- **Logistic Regression Model:** Improved classification over Naive Bayes.
- **BERT Model:** Achieved the highest accuracy after fine-tuning.
- **Confusion Matrix Analysis:** Showed a strong classification performance across all cyberbullying categories.

## Future Work
- Experiment with **transformer models like RoBERTa and XLNet**.
- Implement **real-time cyberbullying detection on social media platforms**.
- Use **data augmentation** to enhance model generalization.

## Citation
If you use this work in your research, please cite:

**Anouska Priya. "Cyber-bullying Types Detection on Twitter: A Comparative Study of Machine Learning Models."**
