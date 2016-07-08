pig1 = [1, 1, 0]
pig2 = [1, 1, 0]
pig3 = [1, 1, 0]
dog1 = [1, 1, 1]
dog2 = [0, 1, 1]
dog3 = [0, 1, 1]

data = [pig1, pig2, pig3, dog1, dog2, dog3]

tags = [1, 1, 1, -1, -1, -1]

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(data, tags)

unknown = [1, 1, 1]
unknown2 = [1, 0, 0]
unknown3 = [1, 0, 1]

tests = [unknown, unknown2, unknown3]

tests_tags = [-1, 1, 1]
result = model.predict(tests)
print(result)

diff = result - tests_tags
hits = [d for d in diff if d == 0]
print(hits)
all_hits = len(hits)
all_elements = len(tests)
hit_tax = 100.0 * all_hits/all_elements
print(hit_tax)
