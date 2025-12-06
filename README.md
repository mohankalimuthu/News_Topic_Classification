# News Topic Classification — Machine Learning Project
This project is all about classifying news articles into different topics using machine learning.
I built it as a simple and clean end-to-end project where the model can read a piece of text and predict the topic it belongs to.

The idea is straightforward:

  > “If I give some news text, can the computer tell whether it is Sports, Business, Politics, Tech, or something else?”

This project answers that question.

## Project Overview
This repository contains everything needed to understand, train, and test a news topic classifier.
It uses TF-IDF for text feature extraction and Multinomial Naive Bayes as the machine-learning model.
You will find these main files:

- training_data.csv — the dataset I used
- news_topic.ipynb — Jupyter Notebook with full code
- topic_classifier_MNB.pkl — trained and saved ML model
- README.md — (this file)

The model takes raw news text as input and returns the predicted category.

## What the Model Does

The model learns patterns from news articles.
For example:
- Sports news usually contains words like “win”, “team”, “match”, “goal”
- Business news has words like “market”, “profit”, “share price”
- Tech news mentions “AI”, “software”, “device”
By learning these patterns, the classifier predicts the correct topic for unseen text.

## Project Structure
'''
📁 News-Topic-Classification

 ├─ training_data.csv
 
 ├─ news_topic.ipynb
 
 ├─ topic_classifier_MNB.pkl
 
 ├─ README.md

'''
## How the Model Works (Simple Steps)

1. Load the dataset
 
      The CSV file contains the news text and their labels.

2. Preprocess the text

     *  Convert to lowercase

     * Remove stopwords

     * Use TF-IDF to convert text into numerical vectors

3. Train the model

     Multinomial Naive Bayes learns which words relate to which topics.
  
4. Test and evaluate

     Accuracy and classification report are generated.

5. Save the model

    The model is saved as topic_classifier_MNB.pkl for later use.

## Technologies Used
 - Python
 - scikit-learn
 - pandas
 - numpy
 - Jupyter Notebook
 - Multinomial Naive Bayes
 - TF-IDF Vectorizer

## How to Run the Project
1. Install Dependencies

        pip install scikit-learn pandas numpy

2. Open the Notebook

        jupyter notebook news_topic.ipynb

3. Run the cells

    This will train the model, test it, and generate evaluation results.

4. Use the Saved Model

You can load the model like this:

      import joblib
      model = joblib.load("topic_classifier_MNB.pkl")

Then predict:

      model.predict(["The stock market crashed due to inflation concerns"])

## Results

The Multinomial Naive Bayes model works very well on text classification because:

 - It handles word frequencies

 - Works best with TF-IDF

 - Fast and lightweight

 - Gives good accuracy for news datasets

Your actual accuracy and metrics will be shown inside the notebook.

## Future Improvements

 Some ideas to make the project even better:

 - Add more cleaning (stemming, lemmatization)

 - Use deep learning models like LSTM or BERT

 - Build a web interface using Flask or Streamlit

 - Add more news categories

 - Create a real-time prediction API
   
