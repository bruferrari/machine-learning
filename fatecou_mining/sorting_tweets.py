# -*- coding: utf-8 -*-

from tt_data import TwitterApi
import yaml

with open('app.yaml', 'r') as ymlfile:
	config = yaml.load(ymlfile)

CONSUMER_KEY = config['api'][0].get('consumer_key')
CONSUMER_SECRET = config['api'][1].get('consumer_secret')
ACCESS_TOKEN = config['api'][2].get('access_token')
ACCESS_TOKEN_SECRET = config['api'][3].get('access_token_secret')

def track(auth):
	track = ['obama', 'hillary']
	stream = twitterApi.get_stream(auth, track)
	print stream

def search(auth, twitterApi):
	query = 'fatec'
	dataframe = twitterApi.search(auth=auth, query=query, count=100)
	print dataframe.head()
	dataframe.to_csv('result.csv', encoding='utf-8')

if __name__ == '__main__':
	twitterApi = TwitterApi()
	auth = twitterApi.authenticate(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
	track(auth)
	# search(auth, twitterApi)