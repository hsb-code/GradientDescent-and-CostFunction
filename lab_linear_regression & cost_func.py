import numpy as np
import matplotlib.pyplot as plt

input_set = [1.0, 2.0]
output_set = [300.0, 500.0]

# data sets
x_train = np.array(input_set)
y_train = np.array(output_set)
print(x_train)
print(y_train)

# number of trains
m = x_train.shape[0]  # total number of rows
print(m)
print(x_train.shape)

# or
m = len(x_train)
print(m)

# access specific training set
i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f'(x^{i}, y^{i}) = ({x_i}, {y_i}) ')

# plot data points
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title('Housing Prices')
plt.xlabel('size(1000 sqft)')
plt.ylabel('price(1000 of dollars)')
plt.show()

# equation of straight line
# f = wx + b
w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")

# calculating straight line


def compute_model_output(x, w, b):
    m = x_train.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i] + b

    return f_wb


tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

w = 200
b = 100
x_i = 1.7
cost_1700sqft = w * x_i + b
print(f'The price of house in is {cost_1700sqft} thousand $')


# to find the value of cost
# w and b must be set to any value that returns cost minimum or closer to minimum
# f_wb = wx + b
# where w and b are paremeters, coeffecients or weights
# you change the values of w and b in order to imrpove model

def compute_cost(x, y, w, b):
    # number of training examples
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost


print(compute_cost(x_train, y_train, 200, 100))
