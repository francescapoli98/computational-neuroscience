#Implement a linear firing rate model (v = w * u) 

#Implement the basic Hebb rule

class HebbianLearning:
    def __init__(self,**kwargs):
        w = random.uniform(low=-1, high=1, size=None)
        v=u*w

    
    def hebbian_rule(self, w, lr):
        w = w + (lr * self.u * self.v) 
        return w
    # def oja_rule():
    #     [...]