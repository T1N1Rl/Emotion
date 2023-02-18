# Emotion
Introduction
1.1	Problem
The aim of the project to understand sentiment of the sentence in terms of the positivity/negativity and neutrality.
Sentiment Analysis is closely connected to Emotion Detection and Recognition from Text, which is a relatively new topic of study. Sentiment Analysis seeks to discover positive, neutral, or negative sentiments in text, whereas Emotion Analysis seeks to detect and distinguish certain types of feelings expressed in text, such as anger, disgust, fear, happiness, sorrow, and surprise. Emotion detection might be used for a variety of purposes, including: 
•	Social media interaction (often to update fresh material on social media)
•	Support (depending on how irritated people are, certain help requests may be more robust than others)
•	Product launch (it may be vital to respond swiftly following the launch of a new product in the event of dissatisfied consumers, bloggers, journalists, etc.

1.2	Literature review
This section highlights different efforts completed by various researchers. In the documentation of Abishek PSS (https://github.com/wendykan/DeepLearningMovies)were employed LSTM model trained on the dataset an LSTM model trained on the dataset was used to predict the sentiment for a specific user review.
The cell state and its many gates are at the heart of LSTMs. The cell state serves as a transportation channel for relative information all the way down the sequence chain. You might think of it as the "memory" of the network. In principle, the cell state can carry meaningful information throughout the sequence's processing. The labeled data set in his study contains 50,000 IMDB movie reviews carefully picked for sentiment analysis. Because review sentiment is binary, an IMDB rating of 5 results in a sentiment score of 0, but a rating of >=7 results in a sentiment score of 1. No single film has gotten more than 30 reviews. The 25,000 review labeled training set contains no videos from the 25,000 review test set. In addition, additional 50,000 IMDB reviews are supplied without rating labels.

Furthermore, Bhagyashree (https://github.com/Bhagya4347/Emotion-Analysis-On-Text) provided several teachniques such as Keyword based detection technique, Lexical Affinity Method, learning based approach, and hybrid methods. Emotions were categorised using Ekman's six fundamental emotions. A set of expressive words is used to represent each emotion group. To assess the emotion represented in a text, its component words were first categorized. The effort began with the extraction of NAVA terms (Adjectives, Nouns, Verbs, and Adverbs) from text. Other words (pronouns, interjections, prepositions, etc.) are ignored since they are always neutral. The data is then clustered, and either unsupervised or supervised learning (SVM) is used to discern the mood of the overall supplied text.

JCharis (https://github.com/Jcharis/end2end-nlp-project) text classification by categorizing 8  emotion (anger, disgust, fear, joy, neutral, sadness, shame, surprise) and compare it to sentiment (positive, neutral, negative).  There were also used typical NLP tokenization, keywords stopping were used by NLTK library.
1.3	Current work
The project goal is to predict finer emotional meaning distinctions based on emotional categories on text, it will focus on the basic task of recognizing emotional passages and determining their valence (positive, neutral, negative) in this study because we do not currently have enough training data to explore finer-grained distinctions. The objective is to gain a thorough grasp of the nature of the NLP problem and to investigate aspects that may be relevant. Same as JCharis we get emotion value count from dataset, implement same NLP techniques: tokenization, stopword removal, keyword extraction, sentiment analysis, wordcloud.
2. Data and Methods
2.1 Information of the data
The first and most critical step in moving any project forward is to collect a good dataset. Books have many different ways of expressing themselves and may be found in almost every language. The empirical research reported in this paper makes use of the great potential for text-based emotion recognition in book chapters. Currently, social media platforms such as Twitter have a lot more information available, therefore I utilized a previously prepared dataset from https://www.kaggle.com/code/akhmadramadani/tweet-emotions-classification.

![image](https://user-images.githubusercontent.com/103248280/219900235-4e0471e9-4edc-4434-bbdf-08834486656c.png)
 
Dataset:
![image](https://user-images.githubusercontent.com/103248280/219900244-c6bcd5ed-38b9-4d1b-83ea-cd7417e6bb23.png)
 
2.2 Description of the ML models
The first stage in analyzing the emotional content of a phrase is to investigate individual words. This is accomplished through a technique known as tokenizing. In my instance, tokens are separated by whitespace characters, such as a space or line break, or punctuation letters. I used the nltk (natural language toolkit) package for the removal of stop words, userhandles, punctuations.
![image](https://user-images.githubusercontent.com/103248280/219900250-4936d72d-706c-41ba-9781-6dee20fca676.png)
 
Next Keyword Extraction is a text analysis NLP tool for quickly gaining valuable insights on a topic. Rather of going through the entire manuscript, the keyword extraction approach may be utilized to condense the content and extract important terms. The keyword Extraction approach is very useful in NLP applications when a company wants to discover customer concerns based on reviews or if you want to find subjects of interest from a recent news item.   
![image](https://user-images.githubusercontent.com/103248280/219900254-c6f519c6-2a40-4245-b502-c2f3b6e29f8b.png)
![image](https://user-images.githubusercontent.com/103248280/219900262-3c1d6370-1958-497a-a2a2-8c3c9a29c3d7.png)

As seen above, the most frequently used terms by Twitter users were extracted. And I plotted the result using Matplotlib and the Seaborn library:
![image](https://user-images.githubusercontent.com/103248280/219900269-a9b7f734-4596-48e8-bb5c-41a5ef42ba6f.png)
Word cloud a text data visualization approach in which each word is shown with its relevance in the context or frequency. This is a very useful program for comprehending the gist of today's news or the content of any YouTube channel. "Concepts are comprehended better with examples".
![image](https://user-images.githubusercontent.com/103248280/219900274-4986ce39-608f-4cec-bb51-4863d5b528e5.png)

Sentiment Analysis using TextBlob
TextBlob is a Python Natural Language Processing package (NLP). Natural Language ToolKit (NLTK) was actively employed by TextBlob to complete its responsibilities. NLTK is a library that provides simple access to a large number of lexical resources and enables users to deal with categorization, classification, and a variety of other tasks. TextBlob is a basic package that allows for extensive textual data analysis and processing.
![image](https://user-images.githubusercontent.com/103248280/219900284-66c2e7e2-5599-41cf-88e2-6c40c5d2f030.png)

3. Results
#Values count of emotions
![image](https://user-images.githubusercontent.com/103248280/219900290-df184062-9f0a-4563-b611-fdf926c45cdf.png)
![image](https://user-images.githubusercontent.com/103248280/219900293-16b26051-4570-4915-bea3-733803b55766.png)

 
#Sentiment for each emotion category
 ![image](https://user-images.githubusercontent.com/103248280/219900294-ec68b2d8-e6f3-4fa0-bcb1-58e6c92540e8.png)    
 ![image](https://user-images.githubusercontent.com/103248280/219900296-25564371-8e53-4583-917b-01d8ec14fb77.png)
![image](https://user-images.githubusercontent.com/103248280/219900300-b790ac42-28b2-448e-b768-7b1174356f29.png)
![image](https://user-images.githubusercontent.com/103248280/219900304-5dd3b893-4958-413e-abdd-3011060305b9.png)

#Removing stopwords, punctuations
 ![image](https://user-images.githubusercontent.com/103248280/219900308-2f7e2c41-1735-4147-999c-7ee8087992b6.png)

#Word Cloud of the anger list
 ![image](https://user-images.githubusercontent.com/103248280/219900312-741e480a-109a-41f8-9890-0f05ccea08fc.png)

#Accuracy and prediction
 ![image](https://user-images.githubusercontent.com/103248280/219900316-2673ba7c-2880-44ee-92a1-0c54e2cb48ea.png)

#Next prediction of emotion from the text
![image](https://user-images.githubusercontent.com/103248280/219900320-4349d1df-9b85-48b4-916b-0c92d22d8cdd.png)
  
#In some cases in output wrong emotion
![image](https://user-images.githubusercontent.com/103248280/219900325-02c2172b-a334-4308-987a-cf55735e257c.png)
 
#Confusion matrix
![image](https://user-images.githubusercontent.com/103248280/219900328-7c004862-fba8-4cb6-aaa4-161f62c648e2.png)
 
4. Discussion
Sentiment analysis, facial recognition, and speech recognition have previously received a lot of research attention. For the time being, I've used a supervised model to identify the emotion in a statement or tweet that is categorized word for word.
Emotions are incompletely defined, and it is particularly uncertain which characteristics may be relevant for recognizing them from text. As a result, we experimented with various feature setups. In certain circumstances, the emotion in the forecast largely detected text as simply one emotion without taking into consideration terms such as "love", "hate", "want", and so on.
Furhermore, In order to establish mature findings, I then want to perform a more thorough analysis using a bigger data set and fix previous mistakes.

References
Abishek PSS, Sentiment Analysis using LSTM, https://medium.com/mlearning-ai/sentiment-analysis-using-lstm-21767a130857
C.O. Alm, D. Roth, R. Sproat, Emotions from text: machine learning for text-based emotion prediction, https://aclanthology.org/H05-1073.pdf
C. Wenyu, H. Nunoo-Mensah, Text-based emotion detection: Advances, challenges, and opportunities, https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.12189
C. Yam, Emotion Detection and Recognition from Text Using Deep Learning, https://devblogs.microsoft.com/cse/2015/11/29/emotion-detection-and-recognition-from-text-using-deep-learning/
R. Ren, Emotion Analysis of Cross-Media Writing Text in the Context of Big Data, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9045008/
S. Bugal, A Stoic Breakdown: Sentiment Analysis, https://caplena.com/blog/sentiment-analysis-explained/

