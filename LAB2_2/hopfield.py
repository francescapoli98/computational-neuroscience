'''
Implement all the code that is required for the different stages in the operation of a
Hopfield network, i.e., the storage phase (learning) and the retrieval phase
(initialization, iteration until convergence, outputting).

Note: in the code you need to include the computation of the overlap functions (with
respect to the training patterns) and of the energy function. 
'''


class Hopfield:
    def __init__(self, **kwargs):
        self.x = x #input_data (the original p0, p1, p2 then distorted into x0, x1, x2)
        self.N =  x.shape[1] #number of neurons (1024s)
        self.p = p #set of patterns
        self.epochs = 100
    
    
    
    @staticmethod
    def storage(self):
        for i,j in self.x:
            if i==j: #fill diagonal as 0
                w=0
            else:
                w=k[:,:,i]@k[:,j,:] #outer product
                w=(np.sum(w,axis=0))/self.N
        return w    
    
    
    
    @property
    def overlap_func(self, p, x):
        for i in p: 
            m=p[:,:,i]@x[:,i,:] #outer product
            m=(np.sum(m,axis=0))/self.N
    
    # @property
    # def energy_func(self, p, x):

        
        
    def __call__(self):
        #LEARNING PHASE
        W = storage() #weight matrix (initialized at 0) 
        newx=np.copy(x)
        #RETRIEVAL PHASE
        for epoch in range(epochs):
            indexes = np.randint(low=0, high=(self.N)-1, size=self.N) #vector of random indexes for state update
            #for each epoch I calculate the overlap functions and the energy function
            #for n in x: #for each set of states correspondent to a pattern in the neurons' set
            for ind in indexes: 
                #get a random neuron and update its state through the update function
                update = W[ind] @ newx[ind]
                
            overlap = overlap_func(self.p, update)
            #energy = energy_func()
        