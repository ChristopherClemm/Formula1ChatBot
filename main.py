import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import csv
import sys
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from tensorflow.python.framework import ops
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import re

#This is the code for the chat bot itself
with open("intents.json") as file:
    data = json.load(file)



words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)
print(output)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")



#code for driver lookups
with open("drivers.csv", encoding="utf8") as csvfile:
    orig = list(csv.reader(csvfile, delimiter=","))



dataDriver = [(item[4].strip().lower()  + " " + item[5].strip().lower()) for item in orig ]

dataDriver = numpy.array(dataDriver)


count = CountVectorizer()
tfidf = TfidfVectorizer(strip_accents = None,
                        lowercase = False,
                        preprocessor = None,
                        use_idf =True,
                        norm='l2',
                        smooth_idf = True)
X = tfidf.fit_transform(dataDriver)

from sklearn.linear_model import LogisticRegression
multiNBDriver = MultinomialNB()
d = [str(i[0]) for i in orig]
multiNBDriver.fit(X, d)


#code for circuit lookups

with open("circuits.csv", encoding="utf8") as csvfile:
    origCircuit = list(csv.reader(csvfile, delimiter=","))



dataCircuit = [(item[4].strip().lower()  + " " + item[5].strip().lower()) for item in origCircuit ]

dataCircuit = numpy.array(dataCircuit)


countCircuit = CountVectorizer()
tfidfCircuit = TfidfVectorizer(strip_accents = None,
                        lowercase = False,
                        preprocessor = None,
                        use_idf =True,
                        norm='l2',
                        smooth_idf = True)
X = tfidfCircuit.fit_transform(dataCircuit)

multiNBCircuit = MultinomialNB()
dCircuit = [str(i[0]) for i in origCircuit]
multiNBCircuit.fit(X, dCircuit)


#load races data
with open("races.csv", encoding="utf8") as csvfile:
    origRaces = list(csv.reader(csvfile, delimiter=","))

#Load results data
with open("results.csv", encoding="utf8") as csvfile:
    origResults = list(csv.reader(csvfile, delimiter=","))


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)





def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tag == 'where' and tg['tag'] == tag:
                where(inp)
                responses = tg['responses']
                break
            elif tag == 'age' and tg['tag'] == tag:
                #print(tg['tag'])
                age(inp)
                responses = tg['responses']
                break
            elif tag == 'nation' and tg['tag'] == tag:
                #print(tg['tag'])
                nationality(inp)
                responses = tg['responses']
                break
            elif tag == 'race winner' and tg['tag'] == tag:
                print(tg['tag'])
                racewinner(inp)
                responses = tg['responses']
                break
            elif tg['tag'] == tag:
                #print("here2")
                #print(tg['tag'])
                responses = tg['responses']
                print(random.choice(responses))

        #print("here")
        #print(random.choice(responses))


def where(inp):
   n = predictCircuit(inp)
   for row in origCircuit:
       if n == row[0]:
           print(row[2] + " is located in " + row[3])



def age(inp):
    n = predictDriver(inp)
    for row in orig:
        if n == row[0]:
            #datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
            print(row[5] + " was born on " + row[6])


def nationality(inp):
    n = predictDriver(inp)
    for row in orig:
        if n == row[0]:
            #datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
            print(row[5] + " is " + row[7])

def racewinner(inp):
    c = predictCircuit(inp)
    year = re.search('\d{4}', imp)
    n = 0
    for row in origRaces:
        if c == row[3] and year == row[1]:
            n = row[0]
    if n == 0:
        print("Sorry I couldn't find the race you wanted, try again")
    else:
        for row in origResults:
            if n == row[0] and row[6] == '1':
                print()



def predictDriver(inp):
    set = {inp.lower()}
    set = tfidf.transform(set)
    n = multiNBDriver.predict(set)
    return n[0]

def predictCircuit(inp):
    setCircuit = {inp.lower()}
    setCircuit = tfidfCircuit.transform(setCircuit)
    n = multiNBCircuit.predict(setCircuit)
    return n[0]


chat()
