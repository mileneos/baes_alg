from math import log
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

mails = pd.read_csv('spam.csv', encoding = 'latin-1')
mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
mails.rename(columns = {'v2': 'message'}, inplace = True)
mails['label'] = mails['v1'].map({'ham': 0, 'spam': 1})
mails.drop(['v1'], axis = 1, inplace = True)
trainIndex, testIndex = list(), list()
for i in range(mails.shape[0]):
    if np.random.uniform(0, 1) < 0.7:
        trainIndex += [i]
    else:
        testIndex += [i]
trainData = mails.loc[trainIndex]
testData = mails.loc[testIndex]
trainData.reset_index(inplace = True)
trainData.drop(['index'], axis = 1, inplace = True)
testData.reset_index(inplace = True)
testData.drop(['index'], axis = 1, inplace = True)

def clean_msg(message):
    message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    sw = stopwords.words('english')
    words = [word for word in words if word not in sw]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return words

class Classifier(object):
    def __init__(self, trainData):
        self.mails= trainData['message']
        self.labels = trainData['label']

    def train(self):
        self.dict_word()
        self.calc_prob()

    def dict_word(self):
        self.spam_mails= self.labels.value_counts()[1]
        self.ham_mails = self.labels.value_counts()[0]
        self.total_mails = self.spam_mails + self.ham_mails
        self.spam_words = 0
        self.ham_words = 0
        self.spam = dict()
        self.ham = dict()
        for i in range(self.mails.shape[0]):
            msg = clean_msg(self.mails[i])
            for word in msg:
                if self.labels[i]:
                    self.spam[word] = self.spam.get(word, 0) + 1
                    self.spam_words += 1
                else:
                    self.ham[word] = self.ham.get(word, 0) + 1
                    self.ham_words += 1

    def calc_prob(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        for word in self.spam:
            self.prob_spam[word] = self.spam[word] / len(list(self.spam.keys()))
        for word in self.ham:
            self.prob_ham[word] = self.ham[word] / len(list(self.ham.keys()))
        self.prob_spam_mail= self.spam_mails / self.total_mails
        self.prob_ham_mail= self.ham_mails / self.total_mails

    def classify(self, msg):
        pSpam, pHam = 0, 0
        for word in msg:
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
            else:
                pSpam -= log(self.spam_words + len(list(self.prob_spam.keys())))
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
            else:
                pHam -= log(self.ham_words + len(list(self.prob_ham.keys())))
            pSpam += log(self.prob_spam_mail)
            pHam += log(self.prob_ham_mail)
        return pSpam >= pHam

    def predict(self, testData):
        result = dict()
        for (i, message) in enumerate(testData):
            msg = clean_msg(message)
            result[i] = int(self.classify(msg))
        return result

def error_prob(labels, predictions):
    count = 0
    for i in range(len(labels)):
        count += int(labels[i] == predictions[i])
    acc = count/len(labels)
    print("Точность прогноза: ", acc)

test = Classifier(trainData)
test.train()
res = test.predict(testData['message'])
error_prob(testData['label'], res)