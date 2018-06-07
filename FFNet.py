import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model


def create_session(memory_fraction=0.1, allow_growth=True):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_growth
    config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    return tf.Session(config=config)
        
class FFNet(object):
    '''An agent class used in testing reinforcement learning algorithms.

    This class is made with the purpose that it would allow multiple agents to
    be trained concurrently in a single game so the majority of their
    work should be hidden behind this class.
    '''

    def __init__(self, model):
        '''Create Agent from model description file.'''
        self.sess = create_session()
        set_session(self.sess)
        with self.sess.as_default(), self.sess.graph.as_default():
            if type(model) is str:
                self.model = load_model(model)
            else:
                self.model = model()
            
    def predict_on_batch(self, batch):
        '''Use Keras api to predict on batch.'''
        with self.sess.as_default(), self.sess.graph.as_default():
            return self.model.predict_on_batch(batch)

    def train_on_batch(self, batch, labels):
        '''Train the Agent.'''
        with self.sess.as_default(), self.sess.graph.as_default():
            return self.model.train_on_batch(batch, labels)
            
    def get_weights(self):
        with self.sess.as_default(), self.sess.graph.as_default():
            return self.model.get_weights()
            
    def set_weights(self, weights):
        with self.sess.as_default(), self.sess.graph.as_default():
            self.model.set_weights(weights)