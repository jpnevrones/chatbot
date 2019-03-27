import json
import nltk
import random
import pickle
import numpy as np
import os
import tflearn
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

def get_intent(intent_file):
    with open(intent_file) as json_data:
        intents = json.load(json_data)
    return intents

def preprocess_data(intents):
    words = []
    classes = []
    documents = []
    ignore_words = ['?']
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))


    classes = sorted(list(set(classes)))

    return documents, classes, words

def get_train_data(classes, documents, words):
    training = []
    output = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    return train_x, train_y

def load_model(root):
    model_data_path = os.path.join(root,"model/model_data.pkl")
    intent_file_path = os.path.join(root,"data/intent.json")
    data = pickle.load(open(model_data_path, "rb"))
    words = data['words']
    classes = data['classes']
    train_x = data['train_x']
    train_y = data['train_y']
    intents = get_intent(intent_file_path)

    # Build neural network
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)

    # Define model and setup tensorboard
    tensorboard_dir = os.path.join(root,"tflearn_logs")
    model = tflearn.DNN(net, tensorboard_dir=tensorboard_dir)
    model_path= os.path.join(root,"model/model.tflearn")
    model.load(model_path)
    return model, words, classes, intents


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))




