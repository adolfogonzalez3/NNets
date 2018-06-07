
from enum import Enum, auto
from queue import Queue
import threading
from threading import Thread

from NNets import FFNet

class FFNetFunctions(Enum):
    predict_on_batch = auto()
    train_on_batch = auto()
    get_weights = auto()
    set_weights = auto()
    Quit = auto()

class Message(object):
    def __init__(self, header, data):
        self.header = header
        self.data = data

def net_async(pipe, args, kwargs):
    mydata = threading.local()
    mydata.net = FFNet(*args, **kwargs)
    while True:
        message = pipe[0].get()
        data = message.data
        if message.header == FFNetFunctions.predict_on_batch:
            pipe[1].put(mydata.net.predict_on_batch(data))
        elif message.header == FFNetFunctions.train_on_batch:
            loss = mydata.net.train_on_batch(data[0], data[1])
        elif message.header == FFNetFunctions.get_weights:
            pipe[1].put(mydata.net.get_weights())
        elif message.header == FFNetFunctions.set_weights:
            mydata.net.set_weights(data)
        elif message.header == FFNetFunctions.Quit:
            break
    
class FFNetAsync(object):
    '''An interface that abstracts the talking with an asynchronous Agent.'''
    def __init__(self, *args, **kwargs):
        parent_conn = child_conn = (Queue(), Queue())
        self.conn = parent_conn
        self.process = Thread(target=net_async, args=(child_conn,
                                                        args, kwargs),
                                                        daemon=True)
        self.process.start()

    def predict_on_batch(self, batch):
        '''Send save message and have the agent save.'''
        self.conn[0].put(Message(FFNetFunctions.predict_on_batch, batch))
        #return self.conn[1].get()
        
    def collect(self):
        return self.conn[1].get()

    def train_on_batch(self, batch, labels):
        '''Send train message and have the agent train asynchronously.'''
        self.conn[0].put(Message(FFNetFunctions.train_on_batch, (batch, labels)))
        #return self.conn[1].get()
        
    def get_weights(self):
        self.conn[0].put(Message(FFNetFunctions.get_weights, None))
        #return self.conn[1].get()
        
    def set_weights(self, weights):
        self.conn[0].put(Message(FFNetFunctions.set_weights, weights))
        
    def close(self):
        self.conn[0].put(Message(FFNetFunctions.Quit, None))
        
    def __del__(self):
        self.close()