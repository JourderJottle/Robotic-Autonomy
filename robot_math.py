import math
import numpy as np

class Gauss2D:

    def __init__(self, u: np.matrix, S: np.matrix) :
        self.u = u
        self.S = S
        self.invS = np.linalg.inv(self.S)
        self.detS = np.linalg.det(self.S)

    def probability(self, x: np.matrix) -> float :
        ud = x - self.u
        return math.exp(-ud.T * self.invS * ud / 2) / math.sqrt(math.pow(2*math.pi, 2) * self.detS)

def rotational_matrix(theta: float) -> np.matrix :
    return np.matrix([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]])

def gauss2D_from_polar(u_d: float, u_theta: float, S: np.matrix) -> Gauss2D :
    R = rotational_matrix(u_theta)
    S_v = R * S * R.T
    return Gauss2D(np.matrix([[u_d * math.cos(u_theta)], [u_d * math.sin(u_theta)]]), S_v)

def local_target_pose_to_global(target_pose: np.matrix, robot_pose: np.matrix, robot_theta: float) -> np.matrix :
    global_target_pose = target_pose + robot_pose
    R = rotational_matrix(robot_theta)
    return R * global_target_pose