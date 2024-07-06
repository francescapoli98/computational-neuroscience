#declaration of the Izhikievich model as a class object

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint

class Izhikievich:
    """The `Izhikievich` class in Python defines a model for simulating neuron behavior using the
    Izhikevich neuron model and provides methods for simulation and plotting membrane potential over time."""
    def __init__(self, tspan, T1, a, b, c, d, V, tau, i, additional = 0, t2=0, t4=0): 
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
        
        self.ii_storing = []
        self.additional = additional
        
        self.T2 = self.T1 + t2
        self.T3 = 0.7 * self.tspan[-1]
        self.T4 = self.T3 + t4
        
    
    
    ######## FUNCTION WITH I #############
    
    def calculate_I(self, t):
        return self.i if t > self.T1+self.additional else 0
    
    
    def simulation(self): 
        V=self.V
        if hasattr(self, 'acc'):
            u=-16
        else:
            u=self.b*V
        VV=[]
        uu=[]
        for t in self.tspan:
            I = self.calculate_I(t)
            if hasattr(self, 'ii'):
                self.ii_storing.append(-90+I)
            # VV, uu = self.update(I, V, u, VV, uu)
            # Reset condition
            if hasattr(self, 'diffV'):
                V = V + self.tau * (0.04 * V ** 2 + 4.1 * V + 108 - u + I)
            else:
                V = V + self.tau * (0.04 * V ** 2 + 5 * V + 140 - u + I)
            if hasattr(self,'acc'):
                u = u + self.tau * self.a * (self.b * (V + 65))    
            else:
                u = u + self.tau * self.a * (self.b * V - u)         
            if V > 30:
                VV.append(30)
                V = self.c
                u = u + self.d
            else:
                VV.append(V)
            uu.append(u)
            if hasattr(self,'acc'):
                self.ii_storing.append(I * 1.5 - 90)
        return VV, uu 
    
    
    ###############   PLOTS   ###############
    
    def x_values(self):
        return [0, self.T1, self.T1, self.T1+3, self.T1+3, max(self.tspan)]
    
    #plot and save membrane potential over time
    def plotting(self, VV, uu, feature_name):
        plt.figure(figsize=(12, 6))
        plt.suptitle(feature_name)

        plt.subplot(2, 1, 1)
        plt.plot(self.tspan, VV, label="Membrane Potential")
        if hasattr(self, 'ii') or hasattr(self,'acc'):
            plt.plot(self.tspan, self.ii_storing, 'k--', label='Input Current')
        else:
            plt.plot([0, self.T1, self.T1, max(self.tspan)], [-90, -90, -80, -80], 'k--')  # Illustrate the input current change
        plt.axis([0, max(self.tspan), -90, 30])
        plt.xlabel("Time span")
        plt.ylabel("Membrane potential")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.title("Phase portrait")
        plt.plot(VV, uu)
        plt.xlabel("Membrane potential")
        plt.ylabel("Recovery variable")
        plt.grid()
        plt.tight_layout()
        plt.savefig(str('plots/'+(feature_name.replace(" ", "_")).lower()+'.jpeg'), edgecolor='black', dpi=400, transparent=True)
        plt.show()
    
    def plotII(self, VV, uu, II, feature_name):
        plt.figure(figsize=(12, 6))
        plt.suptitle(feature_name)
        
        plt.subplot(2, 1, 1)
        plt.plot(self.tspan, VV, label="Membrane Potential")
        plt.xlabel("Time span")
        plt.ylabel("Membrane potential")
        x=self.x_values()
        plt.plot(x, II, 'k--',label="Input Current")  
        if hasattr(self, 'zoom'):
                plt.plot(self.tspan[220:],-10+20*(VV[220:]-np.mean(VV)), label="Zoomed Membrane Potential oscillations")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.title("Phase portrait")
        plt.plot(VV, uu)
        plt.xlabel("Membrane potential")
        plt.ylabel("Recovery variable")
        plt.grid()
        plt.tight_layout()
        plt.savefig(str('plots/'+(feature_name.replace(" ", "_")).lower()+'.jpeg'), edgecolor='black', dpi=400, transparent=True)
        plt.show()
