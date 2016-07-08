#!-*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.cross_validation import cross_val_score 

file_name = 'email.csv'
classifications = pd.read_csv(file_name)
rawTexts = classifications['email']
splitedTexts = rawTexts.str.lower().str.split(' ')

dictionary = set()
for x in splitedTexts:
	dictionary.update(x)

words_count = len(dictionary)
tuples = zip(dictionary, xrange(words_count))

translator = {word:index for word, index in tuples}

def translate(text, translator):
	vector = [0] * len(translator)
	for word in text:
		if word  in translator:
			position = translator[word]
			vector[position] += 1
	return vector

text_vectors = [translate(text, translator) for text in splitedTexts]
tags = classifications['classificacao']

X = np.array(text_vectors)
Y = np.array(tags.tolist())

train_percentage = 0.8
train_lenght = int(train_percentage * len(Y))
validation_length = len(Y) - train_lenght

data_train = X[:train_lenght]
tags_train = Y[:train_lenght]

data_validation = X[train_lenght:]
tags_validation = Y[train_lenght:]

def fit_and_predict(name, model, data_train, tags_train):
    k = 10
    scores = cross_val_score(model, data_train, tags_train)
    hits_tax = np.mean(scores)

    msg = 'Hits tax of model {0}: {1}'.format(name, hits_tax)
    print msg
    return hits_tax

results = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
one_vs_rest_model = OneVsRestClassifier(LinearSVC(random_state = 0))
one_vs_rest_result = fit_and_predict("OneVsRest", one_vs_rest_model, data_train, 
    tags_train)
results[one_vs_rest_result] = one_vs_rest_model

from sklearn.multiclass import OneVsOneClassifier
one_vs_one_model = OneVsOneClassifier(LinearSVC(random_state = 0))
one_vs_one_result = fit_and_predict("OneVsOne", one_vs_one_model, data_train,
    tags_train)
results[one_vs_one_result] = one_vs_one_model

from sklearn.naive_bayes import MultinomialNB
multinomial_model = MultinomialNB()
multinomial_result = fit_and_predict("MultinomialNB", multinomial_model, data_train,
            tags_train)
results[multinomial_result] = multinomial_model

from sklearn.ensemble import AdaBoostClassifier
adaBoost_model = AdaBoostClassifier()
adaBoost_result = fit_and_predict("AdaBoostClassifier", adaBoost_model, data_train,
            tags_train)
results[adaBoost_result] = adaBoost_model

print results
maximum = max(results)
winner = results[maximum]
print 'Winner: {}'.format(winner)

winner.fit(data_train, tags_train)

result = winner.predict(data_validation)
hits = (result == tags_validation)

total_hits = sum(hits)
total_elements = len(tags_validation)
winner_hits_tax = 100.0 * total_hits / total_elements

# Measure hits tax of dummy algorithm for comparison
base_hits = max(Counter(tags_validation).itervalues())
base_hits_tax = 100.0 * base_hits / len(tags_validation)

print('Winner hits tax(in percentage, between both algorithms in real world): {0}'.format(winner_hits_tax))
print 'Number of elements analysed: {0}'.format(len(data_validation)) # Number of analysed elements
print 'Base algorithm hits tax: {0}'.format(base_hits_tax)




















