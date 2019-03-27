import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from chatbot.util import *
import pickle
import numpy as np
import tflearn
import tensorflow as tf
import random
import os

def intentclassifiermodel(train_x, train_y, outpath):
    tf.reset_default_graph()
    # Build neural network
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)

    # Define model and setup tensorboard
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
    # Start training (apply gradient descent algorithm)
    model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
    model_path = os.path.join(outpath,"model.tflearn")
    model.save(model_path)


def train():
    print("Preprocessing the data")
    cwd = os.getcwd()
    print(cwd)
    data_folder = os.path.join(cwd, "data")
    model_folder = os.path.join(cwd, "model")
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    intent_file_path =os.path.join(data_folder, "intent.json")
    model_out_path = model_folder
    intents = get_intent(intent_file_path)
    documents, classes, words = preprocess_data(intents)
    train_x, train_y = get_train_data(classes, documents, words)
    print("Training the model")
    intentclassifiermodel(train_x, train_y, model_out_path)
    model_data_path = os.path.join(model_out_path,"model_data.pkl")
    pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y},
                open(model_data_path, "wb"))
class chatbotRespones():
    def __init__(self, root_dir):
        self.model, self.words, self.classes, self.intents = load_model(root_dir)
        self.context = {}

    def predict(self, sentence, ERROR_THRESHOLD=0.25):
        # generate probabilities from the model
        results = self.model.predict([bow(sentence, self.words)])[0]
        results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.classes[r[0]], r[1]))
        # return tuple of intent and probability
        return return_list

    def get_respones(self, sentence, contentID='123', show_details=False):
        results = self.predict(sentence)
        # if we have a classification then find the matching intent tag
        if results:
            # loop as long as there are matches to process
            while results:
                for i in self.intents['intents']:
                    # find a tag matching the first result
                    if i['tag'] == results[0][0]:
                        # set context for this intent if necessary
                        if 'context_set' in i:
                            if show_details: print('context:', i['context_set'])
                            self.context[contentID] = i['context_set']

                        # check if this intent is contextual and applies to this user's conversation
                        if not 'context_filter' in i or \
                                (contentID in self.context and 'context_filter' in i and i['context_filter'] == self.context[
                                    contentID]):
                            if show_details: print('tag:', i['tag'])
                            # a random response from the intent
                            return random.choice(i['responses'])

                results.pop(0)

if __name__ == '__main__':
    print("Intiliazing  model ")
    cwd = os.getcwd()
    cr = chatbotRespones(cwd)
    sentences = ["Good Morning", "Hello", "What are your hours?", "Do you take credit cards?", "Do you accept Mastercard?", "Goodbye", "see you later", "Good Day"]
    for sentence in sentences:
        prediction = cr.predict(sentence)
        bot_resp = cr.get_respones(sentence)

        print("User: {0}   |   Bot: {1}".format(sentence, bot_resp))
        print("Intent predicted : {0} with probability {1}".format(prediction[0][0],prediction[0][1]))



