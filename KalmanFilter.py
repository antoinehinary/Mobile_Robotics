import random
import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, dt, sigma_q, sigma_r):
        self.x = np.array([[0], [0], [0], [0]]) 
        self.P = np.eye(4) * 25  

        self.F = np.array([[1, dt, 0.5 * dt**2, 0],
                           [0, 1, dt, 0],
                           [0, 0, 1, dt],
                           [0, 0, 0, 1]])

        self.Q = np.array([[sigma_q**2, 0, 0, 0],
                           [0, sigma_q**2, 0, 0],
                           [0, 0, sigma_q**2, 0],
                           [0, 0, 0, sigma_q**2]])

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        self.R = np.array([[sigma_r**2, 0],
                           [0, sigma_r**2]])

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)

        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)

        I = np.eye(self.x.shape[0])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)


qp = 0.04  # variance on position state
rp = 0.25  # variance on position measurement
dt = 0.1
real_positions = []
estimated_positions = []
kf = KalmanFilter(dt, qp, rp)

label_offset = 0.1  
label_fontsize = 12 

for t in range(10):
    control_x = random.randint(-1, 1)
    control_y = random.randint(-1, 1)

    estimate_x = np.array([[control_x], [control_y]])
    estimate_P = np.eye(4) * 1  

    kf.predict()
    kf.update(estimate_x)

    estimated_position = (kf.x[0, 0], kf.x[1, 0])

    real_positions.append((control_x, control_y))
    estimated_positions.append(estimated_position)

plt.plot([position[0] for position in real_positions],
         [position[1] for position in real_positions],
         'r-o', 
         label='Real position')

plt.plot([position[0] for position in estimated_positions],
         [position[1] for position in estimated_positions],
         'b-o', 
         label='Estimated position')

plt.scatter(real_positions[0][0], real_positions[0][1], color='green', marker='s', label='Start (Real)')
plt.scatter(estimated_positions[0][0], estimated_positions[0][1], color='purple', marker='s', label='Start (Estimated)')

for i, (real_pos, est_pos) in enumerate(zip(real_positions[:-1], estimated_positions[:-1])):
    plt.plot([real_pos[0], real_positions[i + 1][0]],
             [real_pos[1], real_positions[i + 1][1]],
             'r-')
    plt.plot([est_pos[0], estimated_positions[i + 1][0]],
             [est_pos[1], estimated_positions[i + 1][1]],
             'b-')
    plt.text(real_pos[0] + label_offset, real_pos[1] + label_offset, str(i + 1), fontsize=label_fontsize, color='red')
    plt.text(est_pos[0] + label_offset, est_pos[1] + label_offset, str(i + 1), fontsize=label_fontsize, color='blue')

# Add labels for the last point
plt.text(real_positions[-1][0] + label_offset, real_positions[-1][1] + label_offset, str(len(real_positions)),
         fontsize=label_fontsize, color='red')
plt.text(estimated_positions[-1][0] + label_offset, estimated_positions[-1][1] + label_offset, str(len(estimated_positions)),
         fontsize=label_fontsize, color='blue')

plt.title("Kalman Filter Position Estimation")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

