# -*- coding: utf-8 -*-

from tt_data import TwitterApi

CONSUMER_KEY = 'put-your-key-here'
CONSUMER_SECRET = 'put-your-key-here'
ACCESS_TOKEN = 'put-your-key-here'
ACCESS_TOKEN_SECRET = 'put-your-key-here'

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
	# track(auth)
	search(auth, twitterApi)