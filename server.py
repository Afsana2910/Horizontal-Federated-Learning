import numpy as np
# Class for the aggregator
class server():
    def __init__( self, theta):
        self.theta = theta

    def receive_from_clients(self, theta1, theta2, theta3):
        W = (theta1[0]+theta2[0]+theta3[0])/3
        b = (theta1[1]+theta2[1]+theta3[1])/3
        self.theta = (W,b)

    def send_to_clients(self):
        return self.theta