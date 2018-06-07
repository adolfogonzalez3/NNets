import unittest
import numpy as np

from create_test_model import create_test_model

import sys
sys.path.append("..")

from NNets import FFNetAsync

class TestFeedForwardNetAsync(unittest.TestCase):

    def test_shape_binary(self):
        model = lambda: create_test_model(10, 1)
        model = FFNetAsync(model)
        model.predict_on_batch(np.ones((1, 10)))
        pred = model.collect()
        self.assertEqual(pred.shape, (1, 1))

    def test_shape_binary_multiple(self):
        model = lambda: create_test_model(10, 1)
        model = FFNetAsync(model)
        model.predict_on_batch(np.ones((10, 10)))
        pred = model.collect()
        self.assertEqual(pred.shape, (10, 1))
        
    def test_shape_multilabel(self):
        model = lambda: create_test_model(10, 10)
        model = FFNetAsync(model)
        model.predict_on_batch(np.ones((1, 10)))
        pred = model.collect()
        self.assertEqual(pred.shape, (1, 10))

    def test_shape_multilabel_multiple(self):
        model = lambda: create_test_model(10, 10)
        model = FFNetAsync(model)
        model.predict_on_batch(np.ones((10, 10)))
        pred = model.collect()
        self.assertEqual(pred.shape, (10, 10))
        
    def test_get_weights(self):
        model = lambda: create_test_model(10, 10)
        model = FFNetAsync(model)
        weights = model.get_weights()
        
    def test_set_weights(self):
        model = lambda: create_test_model(10, 10)
        model_1 = FFNetAsync(model)
        model_2 = FFNetAsync(model)
        weights_1 = model_1.get_weights()
        weights_2 = model_2.get_weights()
        for w_1, w_2 in zip(weights_1, weights_2):
            w_1 = (w_1+w_2)/2
        model_1.set_weights(weights_1)


if __name__ == '__main__':
    unittest.main()