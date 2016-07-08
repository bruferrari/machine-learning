# -*- coding: utf-8 -*-

import tweepy
import pandas as pd
import numpy as np
import json

from datetime import datetime
from pandas.io.json import json_normalize

class StdOutStreamListener(tweepy.StreamListener):

	def on_data(self, data):
		print data
		return True

	def on_status(self, status):
		print status.text 

	def on_error(self, status_code):
		print 'Error: {0}'.format(status_code)
		if status_code == 420:
			# return false if on_data disconnects the stream
			return False

class TwitterApi(object):

	def json_serializer(self, obj):
		""" Json serializer for objects that are not seriablizable by default """

		if isinstance(obj, datetime):
			return obj.isoformat()
		elif obj == None:
			return 'null'
		raise TypeError ('Type is not serializable')

	def authenticate(self, consumer_key, consumer_secret, access_token, secret_token):
		auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
		auth.set_access_token(access_token, secret_token)

		return auth

	def search(self, auth, query, count=30):
		api = tweepy.API(auth)

		tweets = api.search(q=query, count=count)
		
		normalizeds = []
		for tweet in tweets:
			json_str = json.dumps(tweet._json)
			managed_data = {'text': tweet.text, 'lang': tweet.lang, 
							'created_at': tweet.created_at, 'followers': tweet.user.followers_count,
							'screen_name': tweet.user.screen_name, 'geo': tweet.geo}

			managed_data = json.dumps(managed_data, default=self.json_serializer)
			data = json.loads(managed_data)
			normalizeds.append(json_normalize(data))

		result = pd.concat(normalizeds, axis=0)
		print result.head()

		return result

	def get_stream(self, auth, track_arr):
		if isinstance(track_arr, list):
			stream = tweepy.Stream(auth, StdOutStreamListener())
			stream.filter(track=track_arr)
			return stream
		raise TypeError ('track parameter must be an iterable datatype')
	


