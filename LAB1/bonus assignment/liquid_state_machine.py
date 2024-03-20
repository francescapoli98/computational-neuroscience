#Implementing Liquid State Machines (LSMs)
import numpy as np
import matplotlib.pyplot as plt

class LSM:
    def __init__(self,
                Ne=800,
                Ni=200,
                win_e=5,
                win_i=2,
                w_e=0.5,
                w_i=1
                ):
        self.Ne = Ne
        self.Ni = Ni
        re = np.random.rand(Ne)
        ri = np.random.rand(Ni)
        onesNe=np.ones(Ne)
        onesNi=np.ones(Ni)
        v = -65*np.ones(Ne+Ni)  # Initial values of v
        u=b*v
        
    def simulation(self,input):
        U = np.concatenate((win_e*np.ones(Ne), win_i*np.ones(Ni)))
        S = np.concatenate((w_e*np.random.rand(Ne+Ni, Ne), -w_i*np.random.rand(Ne+Ni, Ni)), axis=1)
        firings = []  # spike timings
        states = []  # here we construct the matrix of reservoir states
        
        for t in range(len(input)): 