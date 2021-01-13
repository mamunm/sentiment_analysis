import os 
import tweepy
from tweepy.parsers import JSONParser
import json
from dotenv import load_dotenv, find_dotenv

def fetch_tweets(q=None, 
               lang=None,
               maxResults=None,
               f_name=None):
    
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    # Load the authentication key-value pairs into individual variables
    access_token = os.environ.get("access_token")
    access_token_secret = os.environ.get("access_token_secret")
    consumer_key = os.environ.get("consumer_key")
    consumer_secret = os.environ.get("consumer_secret")
    # Pass OAuth details to tweepy's OAuth handler
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, parser=JSONParser())
    json_str = json.dumps(api.search(q=q, maxResults=maxResults, lang=lang))

    if f_name:
        with open('../data/raw/' + f_name) as f:
            json.dump(json_str)
    else:
        return json_str