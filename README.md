# Emotion Analysis on text

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

JCharis (https://github.com/Jcharis/end2end-nlp-project) text classification by categorizing 8  emotion (anger, disgust, fear, joy, neutral, sadness, shame, surprise) and compare it to sentiment (positive, neutral, negative).  There were also used typical NLP tokenization, words stopping were used by NLTK library.

UTS assessment was employed by Akhmad Ramadani (https://github.com/AkhmadRamadani/learn-machine-learning/blob/6fb6c2d7bb1d5343e55fd036d4439502205f8be0/tweet-emotions-classification.ipynb). He also employed case-folding, tokenization, filtering, and stemming.
1.3	Current work
The project goal is to predict finer emotional meaning distinctions based on emotional categories on text, it will focus on the basic task of recognizing emotional passages and determining their valence (positive, neutral, negative) in this study because we do not currently have enough training data to explore finer-grained distinctions. The objective is to gain a thorough grasp of the nature of the NLP problem and to investigate aspects that may be relevant. Same as JCharis I get emotion value count from dataset, implement same NLP techniques: tokenization, stopword removal, keyword extraction, sentiment analysis, wordcloud.
2. Data and Methods
2.1 Information of the data
The first and most critical step in moving any project forward is to collect a good dataset. Books have many different ways of expressing themselves and may be found in almost every language. The empirical research reported in this paper makes use of the great potential for text-based emotion recognition in book chapters. Currently, social media platforms such as Twitter have a lot more information available, therefore I utilized a previously prepared dataset from https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text?resource=download:
o	Total data: 40000 data
o	Labels: anger, boredom, empty, enthusiasm, fun, happiness, hate, love, neutral, relief, sadness, surprise, worry
o	The amount of data for each label is not the same (class imbalance)
o	There are 3 columns = 'tweet_id', 'emotions', 'content'

![image](https://user-images.githubusercontent.com/103248280/219950744-a7e4e052-b88d-4c8f-93e7-6feacb559388.png)
![image](https://user-images.githubusercontent.com/103248280/219950750-b0c01c94-aa4e-4ab0-aa82-3ac935cbd09b.png)
  
Dataset:
![image](https://user-images.githubusercontent.com/103248280/219950759-db7b5c09-11ae-4abb-a802-a774a4d2dbdd.png)
 
2.2 Description of the ML models
The first stage in analyzing the emotional content of a phrase is to investigate individual words. This is accomplished through a technique known as tokenizing. In my instance, tokens are separated by whitespace characters, such as a space or line break, or punctuation letters. I used the nltk (natural language toolkit) package for the removal of stop words, userhandles, punctuations.
![image](https://user-images.githubusercontent.com/103248280/219950791-d71c54f4-230e-43b6-8ff7-fdc25e86d6f2.png)
 
Next Keyword Extraction is a text analysis NLP tool for quickly gaining valuable insights on a topic. Rather of going through the entire manuscript, the keyword extraction approach may be utilized to condense the content and extract important terms. The keyword Extraction approach is very useful in NLP applications when a company wants to discover customer concerns based on reviews or if you want to find subjects of interest from a recent news item.    
![image](https://user-images.githubusercontent.com/103248280/219950801-d61151dd-731c-4a96-842b-3f1c66af2559.png)
![image](https://user-images.githubusercontent.com/103248280/219950809-3b862de8-4770-4019-9fb2-ce14965dae37.png)

As seen above, the most frequently used terms by Twitter users were extracted. And I plotted the result using Matplotlib and the Seaborn library: 
![image](https://user-images.githubusercontent.com/103248280/219950880-827b792c-9364-43c8-887d-7601835073e2.png)

Word cloud a text data visualization approach in which each word is shown with its relevance in the context or frequency. This is a very useful program for comprehending the gist of today's news or the content of any YouTube channel. "Concepts are comprehended better with examples".
![image](https://user-images.githubusercontent.com/103248280/219950888-43bb396f-ae2d-43ee-82c2-f496ecfa3b83.png)

Sentiment Analysis using TextBlob
TextBlob is a Python Natural Language Processing package (NLP). Natural Language ToolKit (NLTK) was actively employed by TextBlob to complete its responsibilities. NLTK is a library that provides simple access to a large number of lexical resources and enables users to deal with categorization, classification, and a variety of other tasks. TextBlob is a basic package that allows for extensive textual data analysis and processing.
![image](https://user-images.githubusercontent.com/103248280/219950893-072ea6cf-f03a-43b5-ad2d-c4358ca6079b.png)

LSTM modeling
![image](https://user-images.githubusercontent.com/103248280/219950900-6556c1f9-38bc-4a9c-8018-eb697f08105a.png)

3. Results
#Values count of emotions
![image](https://user-images.githubusercontent.com/103248280/219950917-eb8cca09-bf8c-48ea-8317-1eb3a126c6cf.png)
![image](https://user-images.githubusercontent.com/103248280/219950924-f541699e-6f8f-42a4-b455-315ee7271a64.png)
 
#Sentiment for each emotion category
![image](https://user-images.githubusercontent.com/103248280/219950932-17718ded-7da3-42c8-8415-997b69581914.png)
![image](https://user-images.githubusercontent.com/103248280/219950934-82e42392-be91-40d8-9829-9ef6bd294252.png)
![image](https://user-images.githubusercontent.com/103248280/219950940-f8a3598c-7394-4878-8b1b-db314097b32a.png)
![image](https://user-images.githubusercontent.com/103248280/219950942-e810764a-9b08-4f46-a369-31b1eeba58e7.png)
      
#Removing stopwords, punctuations
![image](https://user-images.githubusercontent.com/103248280/219950948-7fa8387d-99d1-400a-840c-df536e8a570c.png)
 
#Word Cloud of the anger list
![image](https://user-images.githubusercontent.com/103248280/219950959-0fede03e-b18f-40b0-96a9-4a9c5ce21650.png)
 
#Accuracy and prediction
![image](https://user-images.githubusercontent.com/103248280/219950975-2c11adca-5f40-40cc-9d9d-0bdbfbaf848e.png)
 
#Next prediction of emotion from the text
![image](https://user-images.githubusercontent.com/103248280/219950987-c4fd1d75-94cb-426b-b5f7-c6ac59a9a368.png)
  
#In some cases in output wrong emotion
![image](https://user-images.githubusercontent.com/103248280/219950991-7d6a47b2-fd5b-4b52-9b4b-118905c1721e.png)
 
#Confusion matrix for visualization of classification
![image](https://user-images.githubusercontent.com/103248280/219950996-42c3283c-bf25-42d7-924f-5e756a19a1d1.png)
 
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

