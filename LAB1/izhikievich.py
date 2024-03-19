#declaration of the Izhikievich model as a class object

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint

# The `Izhikievich` class in Python defines a model with simulation and plotting methods for
# simulating neuron behavior based on the Izhikevich neuron model.
class Izhikievich:
    """docstring for Izhikievich."""
    def __init__(self, tspan, T1, a, b, c, d, V, tau, i): # T1,
        super(Izhikievich, self).__init__()
        self.tspan = tspan    
        self.T1 = T1 
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.V = V
        self.tau = tau
        self.i = i
        
        
    ###################
    def simulation(self): #(tspan,T1,**kwargs): #also to be tested with Euler method
        u=self.b*self.V
        VV=[]
        uu=[]
        #if II, II=[]
        for t in self.tspan:
        # Apply input current based on the condition
            I = self.i if t > self.T1 else 0
            # if II:
            #     II_storing.append(-90+I)
            # Update equations
            V = self.V + self.tau * (0.04 * self.V ** 2 + 5 * self.V + 140 - u + I)
            u = u + self.tau * self.a * (self.b * self.V - u)
            # Reset condition
            if V > 30:
                VV.append(30)
                V = self.c
                u = u + self.d
            else:
                VV.append(V)
            uu.append(u)
            # if II_storing:
            #     return VV, uu, II_storing
            # else:
            return VV, uu
    
    ###############
    #plot and save membrane potential over time
    def plotting(self, VV, uu, feature_name):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 1, 1)
        plt.plot(self.tspan, VV)
        plt.plot([0, self.T1, self.T1, max(self.tspan)], [-90, -90, -80, -80], 'k--')  # Illustrate the input current change
        plt.axis([0, max(self.tspan), -90, 30])
        plt.title(feature_name)
        plt.xlabel("Time span")
        plt.ylabel("Membrane potential")
        plt.savefig(str('membrane potential/'+(feature_name.replace(" ", "_")).lower()+'.jpeg'), edgecolor='black', dpi=400, transparent=True)
        plt.show()