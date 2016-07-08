from data import load_access

X,Y = load_access()

data_train = X[:90]
tags_train = Y[:90]

data_test = X[-9:]
tags_test = Y[-9:]

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(data_train, tags_train)

result = model.predict(data_test)
diff = result - tags_test
hits = [d for d in diff if d == 0]

t_hits = len(hits)
t_elem = len(data_test)
hits_tax = 100.0 * t_hits / t_elem

print(hits_tax)
print "ELEMENTS => %s" %(t_elem)
print "HITS %s" %(t_hits)
