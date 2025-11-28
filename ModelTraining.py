import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
print(data)
print("length of {s} data:", len(data))
labels = np.asarray(data_dict['labels'])
print("length of labels:", len(labels))
v_count = w_count = l_count = 0
for i in range(len(labels)):
    if labels[i] == 0:
        l_count+=1
    elif labels[i] == 1:
        v_count +=1
    else:
        w_count+=1
print(f"v_count: {v_count}, w_count: {w_count}, l_count: {l_count}")

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(data_train, labels_train)

y_predict = model.predict(data_test)

score = accuracy_score(y_predict, labels_test)
print(labels_test)
print(f'{score * 100}% of samples were classified correctly !')

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

