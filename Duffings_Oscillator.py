import numpy as np
import matplotlib.pyplot as plt


# para#import time

#t1 = time.time()  # times the computationmeters (mass = 1)
a = 1                #spring constant
b = 1     #non linear constant
F_0 = 1         #driving force
omega = 1       #driving frequency
gamma = 1  #damping constant
h = 1e-1  # time step
period = 2*np.pi/(1.0*omega)
# length of the simulation
T = 500
t = np.arange(0, T, h)

def derivative(x, v, t):
   
    dxdt = v
    dvdt = -gamma * v + 2.0 * a * x - 4.0 * b * x**3 + F_0 * np.cos(omega * t)
    return dxdt, dvdt

def rk4_step(x, v, t, h):
    k1x, k1v = derivative(x, v, t)
    k2x, k2v = derivative(x + 0.5 * h * k1x, v + 0.5 * h * k1v, t + 0.5 * h)
    k3x, k3v = derivative(x + 0.5 * h * k2x, v + 0.5 * h * k2v, t + 0.5 * h)
    k4x, k4v = derivative(x + h * k3x, v + h * k3v, t + h)
    x_next = x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    v_next = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return x_next, v_next

# initial conditions
v = 0.0
x = 0.0
position = np.zeros(len(t))
velocity = np.zeros(len(t))
position[0] = x

for i in range(1, len(t)):
    x, v = rk4_step(x, v, t[i-1], h)
    position[i] = x
    velocity[i] = v

#poincare
strange_attractor = np.zeros([int(T / period), 2])
k = 1
for i in range(len(t)):
    if abs(t[i] - k * period) < h:
        strange_attractor[k-1, 0] = position[i]
        strange_attractor[k-1, 1] = velocity[i]
        k += 1



# Plot the trajectory of the oscillator
plt.figure(1)
plt.plot(t[-3000:], position[-3000:], 'g-', linewidth=4.0)
plt.title('Trajectory of the oscillator', fontsize=24)
plt.xlabel('Time', fontsize=24)
plt.ylabel('Position', fontsize=24)
plt.tick_params(axis='both', labelsize=24)
plt.grid()

# Plot the phase space
plt.figure(2)
plt.plot(position[-3000:], velocity[-3000:], 'r-', linewidth=4.0)
plt.title('Phase space', fontsize=24)
#plt.xlim([-20, 20])
plt.xlabel('Position', fontsize=24)
plt.ylabel('Momentum', fontsize=24)
plt.tick_params(axis='both', labelsize=24)
plt.grid()

# Plot the Poincare plot
plt.figure(3)
plt.scatter(strange_attractor[:, 0], strange_attractor[:, 1])
plt.xlabel('Position', fontsize=24)
plt.ylabel('Momentum', fontsize=24)
plt.title(r'Poincare Plot (Phase space at time = $\frac{2\pi N}{\omega}$, N = 1,2,3...)', fontsize=24)
plt.tick_params(axis='both', labelsize=24)
plt.grid()

plt.show()
