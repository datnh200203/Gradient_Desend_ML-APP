import numpy as np
import matplotlib.pyplot as plt

x = np.array([  0.245, 0.247, 0.285, 0.299, 0.327, 0.347, 0.356, 0.36, 0.363, 0.364, 
                0.398, 0.4, 0.409, 0.421, 0.432, 0.473, 0.509, 0.529, 0.561, 0.569, 
                0.594, 0.638, 0.656, 0.816, 0.853, 0.938, 1.036, 1.045])
x2 = x**2
y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

def logistic_function(z):
    return 1/(1 + np.exp(-z))

def predict(x, x2, theta0, theta1, theta2):
    z = theta0 + theta1 * x + theta2 * x2
    gz = logistic_function(z)
    return gz

def cost_function(x, x2, y_true, theta0, theta1, theta2):
    m = len(x)
    epsilon = 1e-15
    y_pred = predict(x, x2, theta0, theta1, theta2)
    cost = (-1/m)*(np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)))
    cost_reg = cost + (1/(2*m))*lambda_1*(theta1**2 + theta2**2)
    return cost_reg

def gradient_descent(x, x2, y, theta0, theta1, theta2, learning_rate):
    m = len(x)
    theta0_n = theta0 - learning_rate * (1/m) * np.sum((predict(x, x2, theta0, theta1, theta2) - y))
    theta1_n = theta1*(1 - learning_rate*(lambda_1/m)) - learning_rate * (1/m) * (np.sum((predict(x, x2, theta0, theta1, theta2) - y) * x))
    theta2_n = theta2*(1 - learning_rate*(lambda_1/m)) - learning_rate * (1/m) * (np.sum((predict(x, x2, theta0, theta1, theta2) - y) * x2))
    theta0 = theta0_n
    theta1 = theta1_n
    theta2 = theta2_n
    return theta0, theta1, theta2

lap = 1000
learning_rate = 1e-9
lambda_1 = 0.1 
np.random.seed()
theta0 = np.random.rand()
theta1 = np.random.rand()
theta2 = np.random.rand()

for i in range(0, lap):
    theta0, theta1, theta2 = gradient_descent(x, x2, y, theta0, theta1, theta2, learning_rate)

cost = cost_function(x, x2, y, theta0, theta1, theta2)

print(theta0, theta1, theta2, cost)

# Plotting the data
plt.scatter(x, y, color = 'red', label = 'Data points')
xp = np.linspace(min(x), max(x), 100)
x2p = xp ** 2
yp = predict(xp, x2p, theta0, theta1, theta2)
# Adding labels and title
plt.plot(xp, yp, color = 'blue', label = 'Regression line')
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Sample Plot')

# Display the plot
plt.show()