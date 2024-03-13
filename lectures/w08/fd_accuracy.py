import numpy as np
import matplotlib.pyplot as plt

# Convenience functions for F(x) and F'(x)
def F(x):
    return np.exp(-x ** 2)

def F_derivative(x):
    return -2.0 * x * F(x)

# Test different values of the step size
x = 0.6
# dx = np.array([0.04, 0.02, 0.01, 0.005])
dx = np.logspace(-2, -40, 39, base=2)

# Calculate the FD approximation for all step sizes at once
F_derivative_approx = (F(x + dx) - F(x)) / dx

# Calculate the absolute error
F_derivative_error = np.abs(F_derivative_approx - F_derivative(x))

# Plot the results
fig, ax = plt.subplots()
ax.plot(dx, F_derivative_error, "kx", label='forward')

# Label and tidy up the plot
ax.set(xlabel=r"$\Delta x$", ylabel=r"$F' \left( x \right)$ error magnitude", title="Forward difference")
# ax.set_xlim([0.0, dx.max() * 1.1])
# ax.set_ylim([0.0, 0.01])
ax.set(xscale='log', yscale='log')


# Centred difference
# Calculate the FD approximation for all step sizes at once
F_derivative_approx = (F(x + dx) - F(x - dx)) / (2 * dx)

# Calculate the absolute error
F_derivative_error = np.abs(F_derivative_approx - F_derivative(x))
ax.plot(dx, F_derivative_error, "ro", label='centred')
ax.legend()
plt.show()


h = 1e-10
print(((1+h) - 1) / h)
# print(1+h)