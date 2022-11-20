import numpy as np
from ML import model
class participant():
    def __init__(self,model:model,data):
        self.data = data
        self.model = model(data=self.data)
        

    def receive_from_server(self,theta):
        self.theta = theta

    def train(self):
        self.theta = self.model.fit(self.theta)
        

    def send_to_server(self):
        return self.theta

    
     



    

    

    