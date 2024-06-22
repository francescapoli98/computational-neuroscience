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
    
    
    def simulation(self): #(tspan,T1,**kwargs): #also to be tested with Euler method
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
        plt.subplot(1, 1, 1)
        plt.plot(self.tspan, VV, label="Membrane Potential")
        if hasattr(self, 'ii') or hasattr(self,'acc'):
            plt.plot(self.tspan, self.ii_storing, 'k--', label='Input Current')
        else:
            plt.plot([0, self.T1, self.T1, max(self.tspan)], [-90, -90, -80, -80], 'k--')  # Illustrate the input current change
        plt.axis([0, max(self.tspan), -90, 30])
        plt.title(feature_name)
        plt.xlabel("Time span")
        plt.ylabel("Membrane potential")
        plt.savefig(str('plots/'+(feature_name.replace(" ", "_")).lower()+'.jpeg'), edgecolor='black', dpi=400, transparent=True)
        plt.show()
        
    def plotII(self, VV, uu, II, feature_name):
        plt.figure(figsize=(12, 6))
        plt.plot(self.tspan, VV, label="Membrane Potential")
        plt.title(feature_name)
        plt.xlabel("Time span")
        plt.ylabel("Membrane potential")
        x=self.x_values()
        plt.plot(x, II, 'k--',label="Input Current")  
        # plt.plot([0, self.T1, self.T1, self.T1+self.additional, self.T1+self.additional, max(self.tspan)], II, 'k--',label="Input Current")  
        # [0, T1, T1, T1+8, T1+8, T2, T2, T2+8, T2+8, T3, T3, T3+8, T3+8, T4, T4, T4+8, T4+8, max(tspan)]  
        if hasattr(self, 'zoom'):
                plt.plot(self.tspan[220:],-10+20*(VV[220:]-np.mean(VV)), label="Zoomed Membrane Potential oscillations")
        plt.savefig(str('plots/'+(feature_name.replace(" ", "_")).lower()+'.jpeg'), edgecolor='black', dpi=400, transparent=True)
        plt.show()
        
        
        
        # def simulation_and(self): #(tspan,T1,**kwargs): #also to be tested with Euler method
    #     V=self.V
    #     u=self.b*V
    #     VV=[]
    #     uu=[]
    #     for t in self.tspan:
    #         I = self.i if t > self.T1 and t < self.T1+self.additional else 0
    #         if hasattr(self, 'ii'):
    #             self.ii_storing.append(-90+I)
    #         # VV, uu = self.update(I, V, u, VV, uu)
    #         # Reset condition
    #         V = V + self.tau * (0.04 * V ** 2 + 5 * V + 140 - u + I)
    #         u = u + self.tau * self.a * (self.b * V - u)         
    #         if V > 30:
    #             VV.append(30)
    #             V = self.c
    #             u = u + self.d
    #         else:
    #             VV.append(V)
    #         uu.append(u)
    #     return VV, uu    
        

        
        
        
        ############ CHANGE CALCULATION OF I


#     def default_calculate_I(self, t):
#         return self.i if self.T1 < t < self.T1 + self.additional else 0

#     def simulation_and(self):
#         V = self.V
#         u = self.b * V
#         VV = []
#         uu = []

#         for t in self.tspan:
#             I = self.calculate_I(t)


# # Esempio di uso

# # Definire una nuova funzione per calcolare I
# def custom_calculate_I(self, t):
#     return self.i * 2 if t % 2 == 0 else self.i / 2

# # Creare un'istanza della classe
# instance = MyClass(V=0, tspan=range(100), T1=20, additional=10, tau=0.5, a=0.02, b=0.2, c=-65, d=2, i=5)

# # Modificare il metodo di calcolo di I
# import types
# instance.calculate_I = types.MethodType(custom_calculate_I, instance)

# # Eseguire la simulazione
# VV, uu = instance.simulation_and()
# print(VV, uu)



###################################################################
    
    # def simulation(self): #(tspan,T1,**kwargs): #also to be tested with Euler method
    #     V=self.V
    #     u=self.b*V
    #     VV=[]
    #     uu=[]
    #     for t in self.tspan:
    #     # Apply input current based on the condition            
    #         if hasattr(self, 'i_coef'):
    #             I = self.i + (0.015 * (t-self.T1)) if t > self.T1 else self.i
    #         else:
    #             I = self.i if t > self.T1+self.additional else 0
                
    #         if hasattr(self, 'ii'):
    #             self.ii_storing.append(-90+I)
    #         # Reset condition
    #         V = V + self.tau * (0.04 * V ** 2 + 5 * V + 140 - u + I)
    #         u = u + self.tau * self.a * (self.b * V - u)
    #         # VV, uu = self.update(V, u, VV, uu)
    #         if V > 30:
    #             VV.append(30)
    #             V = self.c
    #             u = u + self.d
    #         else:
    #             VV.append(V)
    #         uu.append(u)
    #     return VV, uu

    
    # def class1ex(self):
    #     V=self.V
    #     u=self.b*V
    #     VV=[]
    #     uu=[]
    #     II = []
    #     for t in self.tspan:
    #         I = self.i * (t-self.T1) if t > self.T1 else 0
    #         # I = inputcurrent(t)
    #         # II.append(-90+I)
    #         self.ii_storing.append(-90+I)
    #         # Update equations
    #         V = V + self.tau * (0.04 * V ** 2 + 4.1 * V + 108 - u + I)
    #         u = u + self.tau * self.a * (self.b * V - u)
    #         # Reset condition
    #         if V > 30:
    #             VV.append(30)
    #             V = self.c
    #             u = u + self.d
    #         else:
    #             VV.append(V)
    #         uu.append(u)
    #     return VV, uu
    
    