from collections import Counter
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
df = pd.read_csv('situation.csv')

X_df = df[['recencia', 'frequencia', 'semanas_de_inscricao']]
Y_df = df['situacao']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

train_percentage = 0.8
train_lenght = int(train_percentage * len(Y))

data_train = X[:train_lenght]
tags_train = Y[:train_lenght]

test_percentage = 0.1
test_lenght = test_percentage * len(Y)

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

t_hits = sum(hits)
t_elements = len(tags_validation)
w_hits_tax = 100.0 * t_hits / t_elements

print('Winner hits tax(in percentage, between both algorithms in real world): {0}'.format(w_hits_tax))
print 'Number of elements analysed: %d' %(len(data_validation)) # Number of analysed elements

