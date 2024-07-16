import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

import torch
from torch import cuda, zeros, Tensor, optim
from torch.nn import Module, ModuleList, Sequential, Linear, Tanh, MSELoss, RNN



# This class `GS` implements a grid search algorithm to find the best model configuration for either a
# TDNN or RNN neural network based on specified parameters and datasets.
class GS:
    
    def __init__(self, parameters:dict, Xset:tuple, Yset:tuple, neuralnet:str):
        self.Xset=Xset
        self.Yset=Yset
        param_grid = self.grid(parameters) #self, 
        min_loss, best_model = self.search(param_grid, neuralnet) #self, 
        self.min_loss = min_loss
        self.best_model = best_model # the best model discovered

    # @staticmethod
    def grid(self, params):
        param_names=list(params.keys())
        param_values=list(params.values())
        param_combinations=list(itertools.product(*param_values))
        
        param_grid=[]
        for combination in param_combinations:
            param_grid.append(dict(zip(param_names, combination)))
        return param_grid 
        
    @staticmethod
    def get_optimizer(model, optimizer_name, lr):
        if optimizer_name == 'adam':
            return optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError("Unknown optimizer: {}".format(optimizer_name))
        
    # @staticmethod
    def search(self, param_grid:tuple, neuralnet:str):#, Xset, Yset):
        predictions={}  
        for pg in param_grid:
            if neuralnet=='TDNN':
                model=TDNN_model(
                    window=pg['window'],
                    layers=pg['hiddens'],
                    input_dim=self.Xset.shape[0],
                    hidden_dim=10, # CONSIDER GRID SEARCHING THIS ALSO
                    #   THOSE TO BE PASED TO .forward AND .train METHODS
                    epochs=pg['epochs']       
                )
            elif neuralnet=='RNN':
                model=RNN_model(
                    input_dim=self.Xset.size(0), #self.Xset.shape[1]
                    hidden_dim=10, # CONSIDER GRID SEARCHING THIS ALSO
                    layers=pg['hiddens'],
                    #   THOSE TO BE PASED TO .forward AND .train METHODS
                    epochs=pg['epochs'],
                    #lr=pg['lr'],
                    #opt=pg['opt']  
                    # optim = 
                ) 
            print("MODEL:", model)   
            optimizer = self.get_optimizer(model, pg['opt'], pg['lr'])      
            loss, y_pred=model.train(self.Xset, self.Yset, optimizer)# pg['lr']) #pg['opt'],
            predictions[loss]=[pg, y_pred] 
        
        
        # predictions=dict(sorted(predictions.items()))
        # print(next(iter(predictions.items())))
        # min_loss=list(predictions.keys())[0]
        min_loss = min(predictions.keys())
        # best_model=list(predictions.values())[0]
        best_model = predictions[min_loss]
        
        return min_loss, best_model
    
    
    