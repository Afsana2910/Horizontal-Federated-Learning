import numpy as np
# Multivariate Linear Regression Model
class model():
      
    def __init__( self, data, learning_rate=0.001, iterations=10) :
        self.x = data[0]
        self.y = data[1]
        # self.w = theta[0]
        self.w =  np.zeros((self.x.shape[1],1))
        self. b = 0
        self.learning_rate = learning_rate
          
        self.epochs = iterations
          
              
    def loss(self):
        cost = np.sum((((self.x.dot(self.w) + self.b) - self.y) ** 2) / (2*len(self.y)))
        return cost

    def fit(self,theta):
        self.w = theta[0]
        self.b = theta[1]
        cost_list = [0] * self.epochs
    
        for epoch in range(self.epochs):
            z = self.x.dot(self.w) + self.b
            loss = z - self.y
            
            weight_gradient = self.x.T.dot(loss) / len(self.y)
            bias_gradient = np.sum(loss) / len(self.y)
            
            self.w = self.w - self.learning_rate*weight_gradient
            self.b = self.b - self.learning_rate*bias_gradient
    
            cost = self.loss()
            cost_list[epoch] = cost
            
            # if (epoch%(self.epochs/10)==0):
            #     print("Cost is:",cost)
            
        return self.w, self.b