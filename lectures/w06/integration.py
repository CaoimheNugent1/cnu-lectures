# Module for composite numerical integration using Riemann sums,
# the midpoint rule, or the trapezoid rule.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def test_after_import(x):
    '''
    This is a toy function to see what happens when importing this module.
    '''
    print(f'Import successful! x is {x}.')


def get_nodes(rule, a, b, N):
    '''
    Return a NumPy array of node locations for a given composite integration
    rule, where the interval [a, b] is divided into N segments.

    rule must be either 'riemann_L', 'riemann_R', 'midpoint', or 'trap'.
    '''
    if rule not in ['riemann_L', 'riemann_R', 'midpoint', 'trap']:
        raise ValueError(f'\'{rule}\' is not a valid rule!')

    # Width of each sub-interval, and total number of nodes
    h = (b - a) / N
    N_nodes = N

    # Define the first and last node positions
    if rule == 'riemann_L':
        # Nodes are the lower bound (left) of each sub-interval
        first_node = a
        last_node = b - h
    elif rule == 'riemann_R':
        # Nodes are the upper bound (right) of each sub-interval
        first_node = a + h
        last_node = b
    elif rule == 'midpoint':
        # Nodes are the centre of each sub-interval
        first_node = a + 0.5*h
        last_node = b - 0.5*h
    elif rule == 'trap':
        # 2 nodes per sub-interval, overlapping at boundaries; total N+1 nodes
        first_node = a
        last_node = b
        N_nodes += 1
    
    # Create the array of nodes
    x_node = np.linspace(first_node, last_node, N_nodes)
    return x_node


def get_vertices(rule, x_node, f_node):
    '''
    Get vertex coordinates for the purpose of illustrating a given integration rule.
    '''
    # Start an empty list to store all sets of vertex coordinates
    verts = []

    # Get number of sub-intervals and sub-interval width
    h = x_node[1] - x_node[0]
    N = len(x_node)
    if rule == 'trap':
        N -= 1

    # Get coordinates for each rectangle or trapezoid
    for k in range(N):
        # bottom left, bottom right, top right, top left
        if rule == 'riemann_L':
            BL = [x_node[k], 0]
            BR = [x_node[k] + h, 0]
            TR = [x_node[k] + h, f_node[k]]
            TL = [x_node[k], f_node[k]]
        elif rule == 'riemann_R':
            BL = [x_node[k] - h, 0]
            BR = [x_node[k], 0]
            TR = [x_node[k], f_node[k]]
            TL = [x_node[k] - h, f_node[k]]
        elif rule == 'midpoint':
            BL = [x_node[k] - 0.5*h, 0]
            BR = [x_node[k] + 0.5*h, 0]
            TR = [x_node[k] + 0.5*h, f_node[k]]
            TL = [x_node[k] - 0.5*h, f_node[k]]
        elif rule == 'trap':
            BL = [x_node[k], 0]
            BR = [x_node[k] + h, 0]
            TR = [x_node[k] + h, f_node[k+1]]
            TL = [x_node[k], f_node[k]]

        # Append the list of vertices
        verts.append([BL, BR, TR, TL])

    return verts


def display_diagram(rule, f, a, b, N):
    '''
    Display a diagram illustrating a given composite integration rule,
    using an example function f plotted over [a, b],
    dividing the interval [a, b] into N segments.
    
    rule must be either 'riemann_L', 'riemann_R', 'midpoint', or 'trap'.
    '''
    # Create an x-axis with 100 points and estimate the function
    x_plot = np.linspace(a, b, 100)
    f_plot = f(x_plot)

    # Get the node positions and evaluate the function at each node
    x_node = get_nodes(rule, a, b, N)
    f_node = f(x_node)

    # Plot the function
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_plot, f_plot, 'k-')

    # Plot the rectangles or trapezoids
    verts = get_vertices(rule, x_node, f_node)
    for k in range(N):
        # Draw the polygon
        polyg = patches.Polygon(verts[k], edgecolor='k')
        ax.add_patch(polyg)
        
    # Plot the nodes
    ax.plot(x_node, f_node, 'rx', markersize=8)

    # Label the plot
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')

    ax.set_title(f'Rule: {rule}')

    plt.show()


def estimate_integral(rule, f, a, b, N):
    '''
    Estimate the integral of a function f over [a, b],
    dividing the interval [a, b] into N segments,
    using a specified composite rule.
    
    rule must be either 'riemann_L', 'riemann_R', 'midpoint', or 'trap'.
    '''
    if rule not in ['riemann_L', 'riemann_R', 'midpoint', 'trap']:
        raise ValueError(f'\'{rule}\' is not a valid rule!')

    # Get the width of each interval
    h = (b - a) / N

    # Estimate the integral as a weighted sum
    nodes = get_nodes(rule, a, b, N)
    if rule != 'trap':
        # Riemann sums or midpoint rule
        I_approx = np.sum(h * f(nodes))
    else:
        # Trapezoid rule
        I_approx = np.sum(h * 0.5 * (f(nodes[:-1]) + f(nodes[1:])))

    return I_approx


if __name__ == '__main__':
    test_after_import(np.pi)
