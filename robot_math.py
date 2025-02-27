import math
import numpy as np

class Gauss2D:

    def __init__(self, u_x, u_y, S) :
        self.u = np.matrix([u_x, u_y])
        self.S = S
    
    def init_polar(self, u_d, u_theta, S) :
        self.u = np.matrix([u_d * math.cos(u_theta), u_d * math.sin(u_theta)])
        R = np.matrix([[math.cos(u_theta), -math.sin(u_theta)],[math.sin(u_theta), math.cos(u_theta)]])
        self.S = R * S * R.T

    def probability(self, x, y) :
        invS = np.linalg.inv(self.S)
        detS = np.linalg.det(self.S)
        ud = np.matrix([x, y]) - self.u
        return math.exp(-ud * invS * ud.T / 2) / math.sqrt(math.pow(2*math.pi, 2) * detS)

def gauss2D_from_polar(u_d, u_theta, S) :
    R = np.matrix([[math.cos(u_theta), -math.sin(u_theta)],[math.sin(u_theta), math.cos(u_theta)]])
    S_v = R * S * R.T
    return Gauss2D(u_d * math.cos(u_theta), u_d * math.sin(u_theta), S_v)