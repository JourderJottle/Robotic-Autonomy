import numpy as np

def derive_gradient(func, location, dl) :
    dimensions = len(location)
    j1 = []
    j2 = []
    for i in range(dimensions) :
        dx = np.zeros(dimensions)
        dx[i] = dl / 2
        x1 = location - dx
        x2 = location + dx
        j1.append(func(x1))
        j2.append(func(x2))
    return (np.array(j2) - np.array(j1)) / dl

def predict(previous_state, previous_covariance, input, motion_model, motion_noise, dt) :
    A = derive_gradient(motion_model, input, 0.1)
    return (previous_state + dt * motion_model(input), A @ previous_covariance @ A.T + motion_noise)

def correct(predicted_state, predicted_covariance, observation, sensor_model, sensor_noise) :
    C = derive_gradient(sensor_model, predicted_state, 0.1)
    K = predicted_covariance @ C.T @ np.linalg.inv(C @ predicted_covariance @ C.T + sensor_noise)
    return (predicted_state + K @ (observation - sensor_model(predicted_state)), (np.diag(np.ones(3)) - K @ C) @ predicted_covariance)