
import numpy as np
from create_test_model import create_test_model
from NNets import FFNetAsync
from time import sleep, time


def func(N=1000):
    model = create_test_model(N, 10)
    array = np.random.rand(10000, N)
    labels = np.random.rand(10000, 10)
    while True:
        flag = (yield)
        if flag is True:
            model.predict_on_batch(array)
            yield 0
        else:
            model.predict_on_batch(array)
            yield 1

def task(N=1000):
    mydata = threading.local()
    sess = tf.Session()
    set_session(sess)
    with sess.as_default(), sess.graph.as_default():
        mydata.model = create_test_model(N, 10)
    array = np.random.rand(10000, N)
    labels = np.random.rand(10000, 10)
    for i in range(10):
        with sess.as_default(), sess.graph.as_default():
            x = mydata.model.predict_on_batch(array)
            
if __name__ == '__main__':
    NUM = 10
    gens = [func() for _ in range(NUM)]
    for g in gens:
        g.send(None)
    results = []
    begin = time()
    for i in range(10):
        for g in gens:
            results.append(g.send(False))
            next(g)
        
    print('Time Elapsed: {!s}'.format(time()-begin))
    begin = time()
    for i in range(10):
        for g in gens:
            results.append(g.send(False))
            next(g)
            
    print('Time Elapsed: {!s}'.format(time()-begin))

    array = np.random.rand(10000, 1000)
    f = lambda: create_test_model(1000, 10)
    gens = [FFNetAsync(f) for _ in range(NUM)]
    begin = time()
    for _ in range(10):
        for g in gens:
            g.predict_on_batch(array)
    for _ in range(10):
        results = [g.collect() for g in gens]
    print('Time Elapsed: {!s}'.format(time()-begin))
    
    begin = time()
    for _ in range(10):
        for g in gens:
            g.predict_on_batch(array)
    for _ in range(10):
        results = [g.collect() for g in gens]
    print('Time Elapsed: {!s}'.format(time()-begin))