import numpy as np


# Defined constants
B = np.array([[4, -2], [-2, 4]])
a = np.array([[0], [1]])
b = np.array([[-2], [1]])


def f1(x):
    """ Function f1 taking input x with shape (2, 1) """
    return float(x.T @ B @ x - x.T @ x + a.T @ x - b.T @ x)

def f2(x):
    """ Function f2 taking input x with shape (2, 1) """
    return float(np.cos((x - b).T @ (x - b)) + (x - a).T @ B @ (x - a))

def f3(x):
    """ Function f3 taking input x with shape (2, 1) """
    return float(1 - (np.exp(-(x - a).T @ (x - a)) + \
                 np.exp(-(x - b).T @ B @ (x - b)) - \
                 (1/10.) * np.log(np.linalg.det((1/100.) * np.identity(2) + x @ x.T))))

def f1_check_minimum(B, a, b):
    """ Write a function that returns True if function f1 has a minimum for variables B, a and b, and returns False otherwise.
        Hint: it may not be required to use all B, a and b. """

    # ---- ENTER SOLUTION TO PROBLEM (a) HERE -----
    check = False

    return check

def grad_fd(fn, x, delta=1e-5):
    """ General function that calculates gradient of some 2d function at point x,
        using finite-differences.

    Inputs:
            fn: Function taking input x and returns a scalar
            x: Numpy vector of shape (2, 1)
            delta: Finite-difference delta (epsilon) used for approximation

    Returns: Approximated gradient at point x, in shape (1, 2)
    """

    # ---- ENTER SOLUTION TO PROBLEM (b.1) HERE -----
    dfdx = None
    
    return dfdx

def f1_grad_fd(x):
    """ Return gradient of f1, using finite differences """
    return grad_fd(f1, x)

def f2_grad_fd(x):
    """ Return gradient of f2, using finite differences """
    return grad_fd(f2, x)

def f3_grad_fd(x):
    """ Return gradient of f3, using finite differences """
    return grad_fd(f3, x)

def f1_grad_exact(x):
    """ Return gradient of f1, exactly derived by hand """

    # ---- ENTER SOLUTION TO PROBLEM (b.2) HERE -----
    gradient = None

    return gradient

def f2_grad_exact(x):
    """ Return gradient of f2, exactly derived by hand """

    # ---- ENTER SOLUTION TO PROBLEM (b.2) HERE -----
    gradient = None

    return gradient

def f3_grad_exact(x):
    """ Return gradient of f3, exactly derived by exact """

    # ---- ENTER SOLUTION TO PROBLEM (b.2) HERE -----
    gradient = None

    return gradient

def gradient_descent(fn, grad_fn, start_x=-2.0, start_y=0.0, lr=0.5, n_steps=50):
    """ Function that performs gradient descent.

    Inputs: 
        - fn: Function to minimize
        - grad_fn: Function that returns gradient of the function to minimize
        - start_loc: Initial location
        - lr: The learning rate
        - n_steps: Number of steps

    Returns: Tuple containing:
        - trajectory of found points: a list containing numpy (2, 1) column vectors
        - final minimum point: a numpy (2, 1) column vector
        - the value at the minimum: float
    """

    start_loc = np.array([[start_x], [start_y]])
    trajectory = [start_loc]

    # ---- ENTER SOLUTION TO PROBLEM (c) HERE -----
    trajectory = None
    found_minimum_value = None
    found_minimum_loc = None
    
    return trajectory, found_minimum_loc, found_minimum_value


