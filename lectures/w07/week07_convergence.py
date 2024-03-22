import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', size=16)

# We can use the quadrature() function from the Week 6 tutorial
def quadrature(f, xk, wk, a, b):
    '''
    Approximates the integral of f over [a, b],
    using the quadrature rule with weights wk
    and nodes xk.
    
    Input:
    f (function): function to integrate (as a Python function object)
    xk (Numpy array): vector containing all nodes
    wk (Numpy array): vector containing all weights
    a (float): left boundary of the interval
    b (float): right boundary of the interval
    
    Returns:
    I_approx (float): the approximate value of the integral
        of f over [a, b], using the quadrature rule.
    '''
    # Define the shifted and scaled nodes
    yk = (b - a)/2 * (xk + 1) + a
    
    # Compute the weighted sum
    I_approx = (b - a)/2 * np.sum(wk * f(yk))
    
    return I_approx


# Let's choose an arbitrary function (not a polynomial) with a known integral
def f(x):
    return np.arctan(x)

def F(x):
    '''
    Exact value for the indefinite integral of f(x) = atan(x).
    '''
    return x * np.arctan(x) - 0.5 * np.log(1 + x**2)


# fig, ax = plt.subplots()
# x = np.linspace(-5, 5, 200)
# ax.plot(x, f(x))
# plt.show()

# Simpson's rule
xk = np.array([-1, 0, 1])
wk = np.array([1/3, 4/3, 1/3])

# Choose some interval
a, b = 0, 4

# Choose some values of h
# M = [2**k for k in range(2, 11)]
M_vals = np.logspace(2, 10, 9, base=2, dtype=int)
# print(M)
h_vals = (b - a) / M_vals

# Calculate the integral for all the different
# values of h (or M)
I_approx_vals = []
for M in M_vals:
    # Calculate the integral using the composite rule

    # Calculate bounds of each sub-partition
    c = np.linspace(a, b, M+1)
    # print(c)

    # Sum up integral approximates over sub-intervals
    I_approx = 0
    for i in range(M):
        I_approx += quadrature(f, xk, wk, c[i], c[i+1])
    
    I_approx_vals.append(I_approx)

# Plot log h vs log E
I_exact = F(b) - F(a)
error = np.abs(I_exact - np.array(I_approx_vals))
# print(error)

fig, ax = plt.subplots()
# ax.plot(np.log(h_vals), np.log(error), 'x')
# ax.set(xlabel='log(h)', ylabel='log(error)')

ax.plot(h_vals, error, 'x')
ax.set(xscale='log', yscale='log')
plt.show()

# Find the slope r
coeffs = np.polyfit(np.log(h_vals), np.log(error), 1)
print(f'r = {coeffs[0]}')