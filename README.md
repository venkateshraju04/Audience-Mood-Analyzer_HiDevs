# Audience Mood Analyzer

This project is automatically generated.
## Create a Reddit app
To extract comments from a reddit post you need to first create a reddit preferences app by visiting www.reddit.com/prefs/apps
## Installation
Install all the required libraries
```sh
pip install -r requirements.txt
```
## Create a .env file and add the following variables
```
REDDIT_API_KEY="YOUR_REDDIT_API_KEY"
REDDIT_SECRET_KEY="YOUR_REDDIT_SECRET_KEY"
USER_AGENT="sentiment-analyzer by u/ username"
COHERE_API="YOUR_COHERE_API" 
```

Since it is a bit time consuming to get all APIs and secret keys, I have pushed the .env file also . Enjoy!

Also added a ipynb file to be able to run in colab. uses ngrok for streamlit interface.
