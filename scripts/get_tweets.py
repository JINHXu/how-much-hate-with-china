# get tweets through Twitter's API (this script can only get tweets within the recent 7 days)
# Jinghua Xu

import tweepy
from keys import Keys
from nltk.tokenize import TweetTokenizer
import json
import numpy as np
import gzip
import os
import pandas as pd


cwd = os.getcwd()
keys_path = os.path.join(cwd, 'resources/tokens.json')
keys = Keys(keys_path)
opt_path = os.path.join(cwd, 'data/hate_china.csv')


df = pd.DataFrame(columns=['text', 'source', 'url'])
msgs = []
msg = []

# tweepy initialization
auth = tweepy.OAuthHandler(keys.consumer_key, keys.consumer_secret)
auth.set_access_token(keys.access_token, keys.access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

c = 0


for tweet in tweepy.Cursor(api.search, q='#china since:2022-02-20 until:2022-02-21', exclude_retweets=True, exclude_replies=True, lang="en").items():

    c += 1
    print(tweet.created_at.date())

print(c)

#     msg = [tweet.text, tweet.source, tweet.source_url]
#     msg = tuple(msg)
#     msgs.append(msg)
#     print(msg)

# df = pd.DataFrame(msgs)
# df.to_csv(opt_path)
