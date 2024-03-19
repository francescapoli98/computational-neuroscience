#declaration of the Izhikievich model as a class object

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint

class Izhikievich:
    """docstring for Izhikievich."""
    def __init__(self, a, b, c, d, V, tspan, tau):
        super(Izhikievich, self).__init__()
        self.a = a
    
    
    ###################
    def simulation(tspan,T1,**kwargs): #also to be tested with Euler method
        u=b*V
        VV=[]
        uu=[]
        #if II, II=[]
        for t in tspan:
        # Apply input current based on the condition
            I = i if t > T1 else 0
            if II:
                II_storing.append(-90+I)
            # Update equations
            V = V + tau * (0.04 * V ** 2 + 5 * V + 140 - u + I)
            u = u + tau * a * (b * V - u)
            # Reset condition
            if V > 30:
                VV.append(30)
                V = c
                u = u + d
            else:
                VV.append(V)
            uu.append(u)
            if II_storing:
                return VV, uu, II_storing
            else:
                return VV, uu
    
    ###############
    #plot and save membrane potential over time
    def plotting(tspan,T1,VV,uu,feature_name):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 1, 1)
        plt.plot(tspan, VV)
        plt.plot([0, T1, T1, max(tspan)], [-90, -90, -80, -80], 'k--')  # Illustrate the input current change
        plt.axis([0, max(tspan), -90, 30])
        plt.title(feature_name)
        plt.xlabel("Time span")
        plt.ylabel("Membrane potential")
        plt.savefig(str('membrane potential/'+(feature_name.replace(" ", "_")).lower()+'.jpeg'), edgecolor='black', dpi=400, transparent=True)
        plt.show()