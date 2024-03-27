#Implementing Liquid State Machines (LSMs)
import numpy as np
from numpy import ones
import matplotlib.pyplot as plt

class LSM:
    """The LSM class implements a simulation of a spiking neural network using Leaky Integrate-and-Fire neurons. """
    def __init__(self, epochs:int, units:int, win_e: int, win_i: int, w_e:int, w_i:int):
        self.epochs=kwargs.get('epochs',"10")
        self.units=kwargs.get('units',"100")
        self.win_e=kwargs.get('win_e',"5")
        self.win_i=kwargs.get('win_i',"2")
        self.w_e=kwargs.get('w_e',"0.5")
        self.w_i=kwargs.get('w_i',"1")
        self.Ne=800
        self.Ni=200
        
        
                
    def simulation(self,input):
        self.re = np.random.rand(self.Ne)
        self.ri = np.random.rand(self.Ni)
        self.a = np.concatenate((0.02*ones(self.Ne), 0.02+0.08*self.ri))
        self.b = np.concatenate((0.2*np.ones(self.Ne), 0.25-0.05*self.ri))
        self.c = np.concatenate((-65+15*self.re**2, -65*np.ones(self.Ni)))
        self.d = np.concatenate((8-6*self.re**2, 2*np.ones(self.Ni)))
        self.v = -65*np.ones(self.Ne+self.Ni)  # Initial values of v
        self.u=self.v*self.b
        
        U = np.concatenate((self.win_e*np.ones(self.Ne), self.win_i*np.ones(self.Ni)))
        S = np.concatenate((self.w_e*np.random.rand(self.Ne+self.Ni, self.Ne), -self.w_i*np.random.rand(self.Ne+self.Ni, self.Ni)), axis=1)
        firings = []  # spike timings
        states = []  # here we construct the matrix of reservoir states
        v=self.v
        u=self.u
        for t in range(len(input)):  # simulation of 1000 ms
            I = input[t] * U
            fired = np.where(v >= 30)[0]  # indices of spikes
            firings.append(np.column_stack((t+np.zeros_like(fired), fired)))
            v[fired] = self.c[fired]
            u[fired] = u[fired] + self.d[fired]
            I = I + np.sum(S[:, fired], axis=1)
            v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)  # step 0.5 ms
            v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)  # for numerical stability
            u = u + self.a*(self.b*v - u)
            states.append(v >= 30)

        firings = np.concatenate(firings)
        plt.plot(firings[:, 0], firings[:, 1], '.')

        # in the end states is 1000 x number of time steps
        return states, firings
    
    
    #def plot(self,**kwargs):
        
        