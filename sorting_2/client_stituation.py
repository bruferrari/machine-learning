from collections import Counter
import pandas as pd
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
validation_lenght = int(len(Y) - train_lenght - test_lenght)

test_end = int(train_lenght + test_lenght)
data_test = X[train_lenght:test_end]
tags_test = Y[train_lenght:test_end]

data_validation = X[test_end:]
tags_validation = Y[test_end:]

def fit_and_predict(model_name, model, data_train,
        tags_train, data_test, tags_test):
    model.fit(data_train, tags_train)

    result = model.predict(data_test)
    hits = (result == tags_test)

    t_hits = sum(hits)
    t_elements = len(data_test)
    hits_tax = 100.0 * t_hits / t_elements

    print('{0} hits tax(in percentage): {1}'.format(model_name, hits_tax))
    print('Hits: {0}'.format(t_hits))
    return hits_tax

# mesure efficiency of dummy test algorithm with unique value
def maximum_a_posteriori():
    base_hits = max(Counter(tags_validation).itervalues())
    base_hits_tax = 100.0 * base_hits / len(tags_validation)
    print 'Base algorithm hits tax(in percentage): %f' %(base_hits_tax)
    return base_hits_tax

results = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
one_vs_rest_model = OneVsRestClassifier(LinearSVC(random_state = 0))
one_vs_rest_result = fit_and_predict("OneVsRestClassifier", one_vs_rest_model, data_train, 
    tags_train, data_test, tags_test)
results[one_vs_rest_result] = one_vs_rest_model

from sklearn.multiclass import OneVsOneClassifier
one_vs_one_model = OneVsOneClassifier(LinearSVC(random_state = 0))
one_vs_one_result = fit_and_predict("OneVsOneClassifier", one_vs_one_model, data_train,
    tags_train, data_test, tags_test)
results[one_vs_one_result] = one_vs_one_model

from sklearn.naive_bayes import MultinomialNB
multinomial_model = MultinomialNB()
multinomial_result = fit_and_predict("MultinomialNB", multinomial_model, data_train,
            tags_train, data_test, tags_test)
results[multinomial_result] = multinomial_model

from sklearn.ensemble import AdaBoostClassifier
adaBoost_model = AdaBoostClassifier()
adaBoost_result = fit_and_predict("AdaBoostClassifier", adaBoost_model, data_train,
            tags_train, data_test, tags_test)
results[adaBoost_result] = adaBoost_model

maximum_a_posteriori()

print results
maximum = max(results)
winner = results[maximum]
print 'Winner: {}'.format(winner)

result = winner.predict(data_validation)
hits = (result == tags_validation)

t_hits = sum(hits)
t_elements = len(tags_validation)
w_hits_tax = 100.0 * t_hits / t_elements

print('Winner hits tax(in percentage, between both algorithms in real world): {0}'.format(w_hits_tax))
print 'Number of elements analysed: %d' %(len(data_validation)) # Number of analysed elements
