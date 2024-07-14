''' '''
import numpy as np
import matplotlib.pyplot as plt



# The `HebbianLearning` class implements Hebbian learning rules such as Hebbian, Oja, and subtractive
# normalization.
class HebbianLearning:
    def __init__(self, data_shape, alpha=0, one=0, theta=0.1, th_lr=0.01):
        self.w_init = np.random.uniform(low=-1, high=1, size=(data_shape,)) #np.random.uniform(data_shape, data_shape, size=(2,)) #np.random.uniform(low=-1, high=1, size=(2,))
        self.lr= 0.001
        # self.epochs=200
        self.alpha = alpha #oja rule
        self.one = one #subtractive normalization
        self.theta=theta #bcm
        self.theta_lr=th_lr

    def hebbian_rule(self, u, v, w):
        w = w + (self.lr * u * v) 
        return w
    
    def oja_rule(self, u, v, w):
        w = (v * u) - self.alpha * (np.power(v, 2) * w)
        return w
    
    def sub_norm(self, u, v, w):
        w = v * u - (v * (self.one.T @ u) * self.one) / 2
        return w
    
    def bcm(self, u, v, w):
        # w = np.clip(w, -1e5, 1e5)
        v = w @ u
        self.theta += self.theta_lr * (np.power(v, 2) - self.theta)
        return v * u * (v - self.theta)
    
    def covariance(self, u, v, w):
        v = w @ u
        return v * (u - self.theta)
