''' '''
import numpy as np
import matplotlib.pyplot as plt


''' The `HebbianLearning` class contains methods for implementing Hebbian, Oja and Subtractive Normalization learning rules in a
neural network. '''
class HebbianLearning:
    def __init__(self, alpha=0, one=0, **kwargs):
        self.w_init = np.random.uniform(low=-1, high=1, size=(2,))
        self.lr= 0.001
        self.epochs=200
        self.alpha = alpha #oja rule
        self.one = one #subtractive normalization

    def hebbian_rule(self, u, v, w):
        w = w + (self.lr * u * v) 
        return w
    
    def oja_rule(self, u, v, w):
        w = (v * u) - self.alpha * (np.power(v, 2) * w)
        return w
    
    def sub_norm(self, u, v, w):
        w = v * u - (v * (self.one.T @ u) * self.one) / 2
        return w
    
