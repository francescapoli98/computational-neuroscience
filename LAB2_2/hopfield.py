'''
Implement all the code that is required for the different stages in the operation of a
Hopfield network, i.e., the storage phase (learning) and the retrieval phase
(initialization, iteration until convergence, outputting).

Note: in the code you need to include the computation of the overlap functions (with
respect to the training patterns) and of the energy function. 

Use the 3 input vectors p0, p1 and p2 (from the corresponding csv files in the archive) to train the Hopfield network
'''

import numpy as np
from numpy import einsum, fill_diagonal, random


class Hopfield:
    def __init__(self, img: np.array, original:np.array, patterns: np.array, bias: float):
        self.x = img #input_data (the original p0, p1, p2 then distorted into x0, x1, x2)
        self.orig = original #original image to be reconstructed
        self.N =  self.x.shape[0] #number of neurons (1024)
        self.p = patterns #set of patterns
        self.epochs = 2
        self.bias = bias
    
    
    def storage(self):
        """
        The `storage` function calculates the dot product of a matrix with its transpose, sets the diagonal
        elements to zero, and returns the resulting matrix.
        :return: The `storage` method is returning a matrix `w` that is calculated using the Einstein
        summation convention with the numpy `einsum` function. The matrix `w` is the result of multiplying
        the transpose of `self.p` with `self.p`, and then setting the diagonal elements of the resulting
        matrix to 0.
        """
        w = np.einsum('ij,ik->jk', self.p, self.p)/self.N        #w=((self.p.T@self.p).sum(axis=0))/self.N
        np.fill_diagonal(w, 0)
        return w   
    
    
    def overlap_func(self, x: np.array):
        """
        The function `overlap_func` calculates the overlap of an input array with a set of patterns.

        :return: The function `overlap_func` is returning a list `o` containing the dot product of each
        pattern in `self.p` with the input array `x`, divided by the total number of elements `N` in
        each pattern.
        """
        o = [(pattern @ x / self.N) for pattern in self.p]
        return o
    
    def energy_func(self, x: np.array, w: np.array):
        """
        This function calculates the energy based on input arrays x and w using matrix multiplication.
        
        :return: the energy value calculated based on the input arrays `x` and `w`.
        """
        # e = - x.T @ w @ x / 2 - np.sum(self.bias * x)
        e = - 0.5 * (np.sum((w @ x) * x)) - self.bias * np.sum(x)
        return e
        
        
    def __call__(self):
        """
        This function performs a stochastic update of neuron states over multiple epochs, calculating
        overlap and energy functions at each step.
        :return: three values: a list of overlaps, a list of energies, and the updated state vector x.
        """
        w = self.storage()
        # old_x=None
        new_x=np.copy(self.x)
        states_set=[np.copy(self.x)]
        overlaps = []
        energies = []
        for epoch in range(self.epochs):
            indexes = np.random.permutation(range(self.N)) #vector of random indexes for state update
            #for each epoch I calculate the overlap functions and the energy function
            #for n in x: #for each set of states correspondent to a pattern in the neurons' set
            for ind in indexes: 
                #get a random neuron and update its state through the update function
                update = np.sign(np.dot(w[ind], new_x) + self.bias)
                new_x[ind] = update   
                #overlap
                overlap = self.overlap_func(new_x) 
                #energy
                energy = self.energy_func(new_x, w)
                overlaps.append(overlap)
                energies.append(energy)  
                states_set.append(new_x)
            
                if np.array_equal(self.orig, new_x):
                    return(overlaps,energies, states_set)
        
        return(overlaps,energies, states_set) 
    
