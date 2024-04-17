'''
Implement all the code that is required for the different stages in the operation of a
Hopfield network, i.e., the storage phase (learning) and the retrieval phase
(initialization, iteration until convergence, outputting).

Note: in the code you need to include the computation of the overlap functions (with
respect to the training patterns) and of the energy function. 
'''

import numpy as np
from numpy import einsum, fill_diagonal, random


class Hopfield:
    def __init__(self, input_data: np.array, patterns: np.array):
        self.x = input_data #input_data (the original p0, p1, p2 then distorted into x0, x1, x2)
        self.N =  self.x.shape[1] #number of neurons (1024)
        self.p = patterns #set of patterns
        self.epochs = 100
        self.bias = None #TO BE CHANGED
    
    
    def storage(self):
        """
        The `storage` function calculates the dot product of a matrix with its transpose, sets the diagonal
        elements to zero, and returns the resulting matrix.
        :return: The `storage` method is returning a matrix `w` that is calculated using the Einstein
        summation convention with the numpy `einsum` function. The matrix `w` is the result of multiplying
        the transpose of `self.p` with `self.p`, and then setting the diagonal elements of the resulting
        matrix to 0.
        """
        w=np.einsum('ij,ik->jk', self.p, self.p)         #w=((self.p.T@self.p).sum(axis=0))/self.N
        np.fill_diagonal(w, 0)
        return w   
    
    
    
    def overlap_func(self, x):
        """
        The `overlap_func` function calculates the overlap of input `x` with patterns stored in
        `self.p`.
        
        :param x: It looks like the `overlap_func` function takes two parameters: `self` and `x`. The
        `x` parameter seems to be a value that is used in the function to calculate the overlap of
        patterns stored in `self.p`. The function calculates the overlap of each pattern in `self.p
        :return: The `overlap_func` function is returning a list `o` where each element is the dot
        product of a pattern from `self.p` and the input `x`, divided by the total number of elements
        `N` in the pattern.
        """
        o = [(pattern @ x / self.N) for pattern in self.p.T]
        return o
    
    def energy_func(self, x, w):
        """
        The function calculates the energy based on the input vector x and weight matrix w.
        
        :param x: The `x` parameter seems to represent a vector or matrix. The `@` operator is used for
        matrix multiplication in this context
        :param w: The parameter `w` in the `energy_func` function appears to be a weight matrix. It is
        used in the calculation of the energy of the system based on the input `x`. The `@` operator is
        likely performing matrix multiplication between `x` and `w` in this context
        :return: The function `energy_func` returns the energy value calculated based on the input `x`
        and weight `w`.
        """
        e = - x.T @ w @ x / 2 - sum(self.bias * x)
        return e
        
        
    def __call__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        w = self.storage()
        x=self.x.T
        print("Weight matrix: ", w.shape, "input matrix: ", x.shape)
        overlaps = []
        energies = []
        for epoch in range(self.epochs):
            indexes = np.random.permutation(self.N) #vector of random indexes for state update
            #for each epoch I calculate the overlap functions and the energy function
            #for n in x: #for each set of states correspondent to a pattern in the neurons' set
            for ind in indexes: 
                #get a random neuron and update its state through the update function
                update = np.sign(w[ind] @ x) #ADD BIAS HERE
                x[ind] = update
                #overlap
                self.overlap_func(update) 
                #overlap = self.overlap_func(update)
                #energy
                energy = self.energy_func(update, w)
                
                overlaps.append(overlap)
                energies.append(energy)
        return overlaps, energies, x
                
                
                
                
                