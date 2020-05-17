Data science portfolio by Andrey Lukyanenko
This portfolio is a compilation of notebooks which I created for data analysis or for exploration of machine learning algorithms. A separate category is for separate projects.

Stand-alone projects.
Handwritten digit recognition
This is my own project using image recognition methods in practice. This is a site (also works on mobile) where user can draw a digit, and machine learning models (FNN and CNN) will try to recognize it. After than models can use the drawn digit for training to improve their accuracy. Live version is here. The code can be found here.

Chatbot in telegram
A conversational chatbot in telegram which was created for an honor assignment of nlp course by Higher School of Economics. The main functionality of the bot is to distinguish two types of questions (questions related to programming and others) and then either give an answer or talk using a conversational model.

Kaggle competitions.
Avito demand prediction
Avito demand prediction was a competition on Kaggle where we tried to predict something like demand based on ads content. This competition was very interesting because it had tabular data, texts and images. On the other hand this was the reason the competition was quite difficult. My team reached 131th place and got bronze medal! Here is a link to my solution.

Categorization of purchases
This was a Russian inclass Kaggle competition in the third session of ODS ml course. It sounded interesting and I took part in it reaching a 3rd place. Here is my kaggle kernel with a solution.

Kaggle kernels.
2018 Kaggle ML & DS Survey Challenge
Some time ago Kaggle launched a big online survey for kagglers and now this data is public. There were multiple choice questions and some forms for open answers. Survey received 23k+ respondents from 147 countries. As a result we have a big dataset with rich information on data scientists using Kaggle. In this kernel I compare DS in USA, Russia, India and other countries.

DonorsChoose.org Application Screening
DonorsChoose.org empowers public school teachers from across the country to request much-needed materials and experiences for their students. DonorsChoose.org receives hundreds of thousands of project proposals each year for classroom projects in need of funding. This is a competition on Kaggle where people can create a machine learning model to help this fund with auto-approving of applications. Prizes are given to the authors with the most upvoted kernels. Here is my kernel with extensive EDA, feature engineering and building models. This kernel got 2nd place by the number of votes and I won Google Pixelbook for it!

Avito Demand Prediction Challenge
Avito challenge is about predicting demand for an online advertisement based on its full description (title, description, images, etc.), its context (geographically where it was posted, similar ads already posted) and historical demand for similar ads in similar contexts. The competition is interesting due to many types of data in it which allows to build various models. Here is my kernel with EDA, creating features and building models.

Home Credit Default Risk
Home Credit Bank offers a challenge of credit scoring. There is a lot of data about applicants and their previous behavior. Here is my kernel.

Movie Review Sentiment Analysis
Some time ago Kaggle has launched several "remakes" of old competitions. It means that datasets are the same, but now we are offered an opportunity to simply explore the data and create kernels with new methods. One of these competitions is sentiment analysis of Rotten Tomatoes dataset with 5 classes (negative, somewhat negative, neutral, somewhat positive, positive). I have created a kernel with EDA and modern NN architecture: LSTM-CNN. Currently this kernel shows the 5th result of leaderboard.

Two Sigma: Using News to Predict Stock Movements
In this competition Reuters provide unique data, which can't be obtained outside of this competition. We can see a 10 years worth of news and market data on many companies. This competition is kernel-only, which means that everyone has the same amount of computational power for this competition. In my kernel I have analysed the data and showed trends of market data.

Santander Value Prediction Challenge
In this competition we got an anonymized dataset, later it was found that it had a certain structure. In my kernel I tried to analyze the data and created new features using NN model.

Google Analytics Customer Revenue Prediction
RStudio hosted this competition to prove that machine learning algorithms can impact business and help marketing. In my kernel I did an extensive EDA and build an interesting LGB model.

Data Science for Good: Center for Policing Equity
This dataset was provided by The Center for Policing Equity. They hope that kagglers will help to create better models, find some unique insights and improve geo-analytics. In my kernel I try to do such things.

Classification problems.
Titanic: Machine Learning from Disaster
Github nbviewer

Titanic: Machine Learning from Disaster is a knowledge competition on Kaggle. Many people started practicing in machine learning with this competition, so did I. This is a binary classification problem: based on information about Titanic passengers we predict whether they survived or not. General description and data are available on Kaggle. Titanic dataset provides interesting opportunities for feature engineering.

Ghouls, Goblins, and Ghosts... Boo!
Github nbviewer

Ghouls, Goblins, and Ghosts... Boo! is a knowledge competition on Kaggle. This is a multiple classification problem: based on information about monsters we predict their types. A fun competition for Halloween. General description and data are available on Kaggle. This dataset has little number of samples, so careful feature selection and model ensemble are necessary for high accuracy.

Otto Group Product Classification Challenge
Github nbviewer

Otto Group Product Classification Challenge is a knowledge competition on Kaggle. This is a multiple classification problem. Based on information about products we predict their category. General description and data are available on Kaggle. The data is obfuscated, so the main questionlies in the selection of the model for prediction.

Imbalanced classes
Github nbviewer

In real world it is common to meet data in which some classes are more common and others are rarer. In case of a serious disbalance prediction rare classes could be difficult using standard classification methods. In this notebook I analyse such a situation. I can't share the data, used in this analysis.

Bank card activations
Github nbviewer

Banks strive to increase the efficiency of their contacts with customers. One of the areas which require this is offering new products to existing clients (cross-selling). Instead of offering new products to all clients, it is a good idea to predict the probability of a positive response. Then the offers could be sent to those clients, for whom the probability of response is higher than some threshold value. In this notebook I try to solve this problem.

Regression problems.
House Prices: Advanced Regression Techniques
Github nbviewer

House Prices: Advanced Regression Techniques is a knowledge competition on Kaggle. This is a regression problem: based on information about houses we predict their prices. General description and data are available on Kaggle. The dataset has a lot of features and many missing values. This gives interesting possibilities for feature transformation and data visualization.

Loan Prediction
Github nbviewer

Loan Prediction is a knowledge and learning hackathon on Analyticsvidhya. Dream Housing Finance company deals in home loans. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. Based on customer's information we predict whether they should receive a loan or not. General description and data are available on Analyticsvidhya.

Caterpillar Tube Pricing
Github nbviewer

Caterpillar Tube Pricing is a competition on Kaggle. This is a regression problem: based on information about tube assemblies we predict their prices. General description and data are available on Kaggle. Dataset consists of many files, so there is an additional challenge in combining the data snd selecting the features.

Natural language processing.
Bag of Words Meets Bags of Popcorn
Github nbviewer

Bag of Words Meets Bags of Popcorn is a sentimental analysis problem. Based on texts of reviews we predict whether they are positive or negative. General description and data are available on Kaggle. The data provided consists of raw reviews and class (1 or 2), so the main part is cleaning the texts.

NLP with Python: exploring Fate/Zero
Github nbviewer

Natural language processing in machine learning helps to accomplish a variety of tasks, one of which is extracting information from texts. This notebook is an overview of several text exploration methods using English translation of Japanese light novel "Fate/Zero" as an example.

NLP. Text generation with Markov chains
Github nbviewer

This notebook shows how a new text can be generated based on a given corpus using an idea of Markov chains. I start with simple first-order chains and with each step improve model to generate better text.

NLP. Text summarization
Github nbviewer

This notebook shows how text can be summarized choosing several most important sentences from the text. I explore various methods of doing this based on a news article.

Clustering
Clustering with KMeans
Github nbviewer

Clustering is an approach to unsupervised machine learning. Clustering with KMeans is one of algorithms of clustering. in this notebook I'll demonstrate how it works. Data used is about various types of seeds and their parameters. It is available here.

Neural networks
Feedforward neural network with regularization
Github nbviewer

This is a simple example of feedforward neural network with regularization. It is based on Andrew Ng's lectures on Coursera. I used data from Kaggle's challenge "Ghouls, Goblins, and Ghosts... Boo!", it is available here.

Data exploration and analysis
Telematic data
Github nbviewer

I have a dataset with telematic information about 10 cars driving during one day. I visualise data, search for insights and analyse the behavior of each driver. I can't share the data, but here is the notebook. I want to notice that folium map can't be rendered by native github, but nbviewer.jupyter can do it.

Recommendation systems.
Collaborative filtering
Github nbviewer

Recommenders are systems, which predict ratings of users for items. There are several approaches to build such systems and one of them is Collaborative Filtering. This notebook shows several examples of collaborative filtering algorithms.
































<b>Data science portfolio by Dileep.A</b>

This portfolio is a compilation of notebooks which I created for data analysis or for exploration of machine learning algorithms. A separate category is for separate projects.

<b>Stand-alone projects.,</b>
Handwritten digit recognition
This is my own project using image recognition methods in practice. This is a site (also works on mobile) where user can draw a digit, and machine learning models (FNN and CNN) will try to recognize it. After than models can use the drawn digit for training to improve their accuracy. Live version is here. The code can be found here.

<b>Kaggle kernels.</b>

<b>Home Credit Default Risk</b>
Home Credit Bank offers a challenge of credit scoring. There is a lot of data about applicants and their previous behavior. The code can be found here.

<b>Movie Review Sentiment Analysis</b>
Some time ago Kaggle has launched several “remakes” of old competitions. It means that datasets are the same, but now we are offered an opportunity to simply explore the data and create kernels with new methods. One of these competitions is sentiment analysis of Rotten Tomatoes dataset with 5 classes (negative, somewhat negative, neutral, somewhat positive, positive). I have created a kernel with EDA and modern NN architecture: LSTM-CNN. Currently this kernel shows the 5th result of leaderboard.

<b>Data Science for Good: Center for Policing Equity</b>
This dataset was provided by The Center for Policing Equity. They hope that kagglers will help to create better models, find some unique insights and improve geo-analytics. In my kernel I try to do such things.

<b>Classification problems.</b>
Titanic: Machine Learning from Disaster
Azure ML Studio.

Titanic: Machine Learning from Disaster is a knowledge competition on Kaggle. Many people started practicing in machine learning with this competition, so did I. This is a binary classification problem: based on information about Titanic passengers we predict whether they survived or not. General description and data are available on Kaggle. Titanic dataset provides interesting opportunities for feature engineering.

<b>Regression problems.</b>
House Prices: Advanced Regression Techniques
Github nbviewer

House Prices: Advanced Regression Techniques is a knowledge competition on Kaggle. This is a regression problem: based on information about houses we predict their prices. General description and data are available on Kaggle. The dataset has a lot of features and many missing values. This gives interesting possibilities for feature transformation and data visualization.

Loan Prediction
Github nbviewer

Loan Prediction is a knowledge and learning hackathon on Analyticsvidhya. Dream Housing Finance company deals in home loans. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. Based on customer’s information we predict whether they should receive a loan or not. General description and data are available on Analyticsvidhya.

<b>Natural language processing.</b>
Bag of Words Meets Bags of Popcorn
Github nbviewer

Bag of Words Meets Bags of Popcorn is a sentimental analysis problem. Based on texts of reviews we predict whether they are positive or negative. General description and data are available on Kaggle. The data provided consists of raw reviews and class (1 or 2), so the main part is cleaning the texts.

NLP with Python: exploring Fate/Zero
Github nbviewer

Natural language processing in machine learning helps to accomplish a variety of tasks, one of which is extracting information from texts. This notebook is an overview of several text exploration methods using English translation of Japanese light novel “Fate/Zero” as an example.

NLP. Text generation with Markov chains
Github nbviewer

This notebook shows how a new text can be generated based on a given corpus using an idea of Markov chains. I start with simple first-order chains and with each step improve model to generate better text.

NLP. Text summarization
Github nbviewer

This notebook shows how text can be summarized choosing several most important sentences from the text. I explore various methods of doing this based on a news article.

<b>Clustering</b>
Clustering with KMeans
Github nbviewer

Clustering is an approach to unsupervised machine learning. Clustering with KMeans is one of algorithms of clustering. in this notebook I’ll demonstrate how it works. Data used is about various types of seeds and their parameters. It is available here.

Neural networks
Feedforward neural network with regularization
Github nbviewer

This is a simple example of feedforward neural network with regularization. It is based on Andrew Ng’s lectures on Coursera. I used data from Kaggle’s challenge “Ghouls, Goblins, and Ghosts… Boo!”, it is available here.

Data exploration and analysis
Telematic data
Github nbviewer

I have a dataset with telematic information about 10 cars driving during one day. I visualise data, search for insights and analyse the behavior of each driver. I can’t share the data, but here is the notebook. I want to notice that folium map can’t be rendered by native github, but nbviewer.jupyter can do it.

Recommendation systems.
Collaborative filtering
Github nbviewer

Recommenders are systems, which predict ratings of users for items. There are several approaches to build such systems and one of them is Collaborative Filtering. This notebook shows several examples of collaborative filtering algorithms.

