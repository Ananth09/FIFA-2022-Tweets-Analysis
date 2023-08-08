import snscrape.modules.twitter as sntwitter
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

tweets = []

query = '#WorldCup2022 lang:en since:2022-11-20 until:2022-11-21'
q = sntwitter.TwitterSearchScraper(query)

#twittersearchscraper to search tweet and append to list
for i, tweet in enumerate(q.get_items()):
    if i>1000:
        break
    tweets.append([tweet.user.username, tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])

#Convert to dataframe
tweetsdf = pd.DataFrame(tweets, columns=["User", "Date", "Number of Likes", "Source of Tweet", "Tweet"])
tweetsdf.head()
tweetsdf.to_csv('fifa_2022_scrapped.csv', index=False)

#hugging face
sentiment_analysis = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest")

#tweet preprocessing
def preprocess_tweet(row):
    text = row['Tweet']
    text = p.clean(text)
    return text

tweetsdf['Tweet'] = tweetsdf.apply(preprocess_tweet, axis=1)
tweetsdf.head()

tweetsdf['Tweet'] = tweetsdf['Tweet'].str.lower().str.replace('[^\w\s]',' ').str.replace('\s\s+',' ')

tweetsdf.head()
tweetsdf.tail()

tweetsdf.to_csv('fifa_2022_preprocessed.csv', index=False)


#Predicting sentiment
tweetSA = []
for i, tweet in enumerate(q.get_items()):
    if i>30000:
        break
    content = tweet.content
    sentiment = sentiment_analysis(content)
    tweetSA.append({"Date": tweet.date, "Number of Likes": tweet.likeCount,
                     "Source of Tweet": tweet.sourceLabel, "Tweet": tweet.content, 'Sentiment': sentiment[0]['label']})

#converting to dataframe
sadf = pd.DataFrame(tweetSA)
sadf.head()


#sentiment counts
senti_count = sadf.groupby(['Sentiment']).size()
print(senti_count)


#Data Visualization (plotting pie chart)
figure = plt.figure(figsize=(7.5,7.5), dpi=100)
ax = plt.subplot(111)
plt.title(label="FIFA 2022 SENTIMENT ANALYSIS",pad=20)
senti_count.plot.pie(ax=ax, autopct='%1.2f%%', startangle=270, fontsize=12, label="")

#saving file as csv
sadf.to_csv('fifa_2022_analysis.csv', index=False)

