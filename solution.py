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
    # dfdx = x.T @ (B+B.T) - 2*x.T + a.T - b.T
    dfdx2 = (B+B.T).T - 2 

    check = np.sum(dfdx2) >= 0
    # print(np.sum(dfdx2))
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

    # Initiate row grad vector
    partial_derivatives = np.zeros((1, x.shape[0]))
    # Calculate each partial derivatives
    for n in range(x.shape[0]):
        # Only move in the direction of given x_n
        delta_n = np.zeros((x.shape[0], 1))
        delta_n[n,0] = delta
        # Function change in small step of given direction
        numerator = fn(x+delta_n) - fn(x)
        # Input change of given x_n
        denominator = delta
        # Assign partial derivative
        partial_derivatives[0,n] = numerator/denominator

    return partial_derivatives

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
    gradient = x.T @ (B+B.T) - 2*x.T + a.T - b.T
    return gradient

def f2_grad_exact(x):
    """ Return gradient of f2, exactly derived by hand """

    # ---- ENTER SOLUTION TO PROBLEM (b.2) HERE -----
    gradient = None
    gradient = -np.sin((x-b).T @ (x-b)) * 2 * (x-b).T + (x-a).T @ (B+B.T)
    return gradient

def f3_grad_exact(x):
    """ Return gradient of f3, exactly derived by exact """

    # ---- ENTER SOLUTION TO PROBLEM (b.2) HERE -----
    gradient = None
    x1 = x[0,0]
    x2 = x[1,0]


    # X = (1/100.) * np.identity(2) + x @ x.T
    # x_xT_dx = np.array([[ [2*x1, x2 ],
    #                      [x2  ,  0 ] ],
    #                    [ [0   , x1 ],
    #                      [x1  ,2*x2] ]])
    # fxxT_dx = np.trace(np.tensordot(np.linalg.inv(X), x_xT_dx,1))
    # # print(x_xT_dx.shape)
    # # print(np.linalg.inv((1/100.) * np.identity(2) + x @ x.T).T)
    # # grad_third = 1/10. * (np.tensordot(np.linalg.inv((1/100.) * np.identity(2) + x @ x.T).T),  x_xT_dx, 1)
    # grad_first = np.exp(-(x - a).T @ (x - a)) * 2 * (-(x-a).T)
    # grad_second = np.exp(-(x - b).T @ B @ (x - b)) @ (-(x-b).T) @ (B+B.T)
    # grad_third = 1/10*fxxT_dx
    # gradient = - (grad_first+grad_second-grad_third)
    # print(gradient)
    # return gradient

    det_x_xT = 0.01*x1**2+ 0.01*x2**2 + 1/10000 
    det_x_xT_dx1 = 0.02*x1 
    det_x_xT_dx2 = 0.02*x2 
    x_xT_dx = np.array([1/det_x_xT * det_x_xT_dx1 , 1/det_x_xT * det_x_xT_dx2 ])
    grad_first = np.exp(-(x - a).T @ (x - a)) * 2 * (-(x-a).T)
    grad_second = np.exp(-(x - b).T @ B @ (x - b)) @ (-(x-b).T) @ (B+B.T)
    grad_third = 1/10*x_xT_dx

    gradient = - (grad_first+grad_second-grad_third)
    # print(gradient)
    return gradient

def gradient_descent(fn, grad_fn, start_x=-0.5, start_y=0.5, lr=0.05, n_steps=50):
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
    def grad_fd(x, delta=1e-5):
        """ General function that calculates gradient of some 2d function at point x,
            using finite-differences.

        Inputs:
                fn: Function taking input x and returns a scalar
                x: Numpy vector of shape (2, 1)
                delta: Finite-difference delta (epsilon) used for approximation

        Returns: Approximated gradient at point x, in shape (1, 2)
        """

        # ---- ENTER SOLUTION TO PROBLEM (b.1) HERE -----
        dxdf = None
        delta1 = np.array([[delta],[0]])
        delta2 = np.array([[0],[delta]])

        numerator1 = fn(x+delta1) - fn(x)
        numerator2 = fn(x+delta2) - fn(x)

        denominator = delta
        dxdf = np.array([numerator1/denominator, numerator2/denominator])
        
        return dxdf
    # ---- ENTER SOLUTION TO PROBLEM (c) HERE -----

    for i in range(n_steps):
        grad = grad_fn(start_loc)
        new_loc = start_loc - (lr * grad.T)
        trajectory.append(new_loc)
        start_loc = new_loc

    found_minimum_value = fn(new_loc)
    found_minimum_loc = new_loc
    
    return trajectory, found_minimum_loc, found_minimum_value


if __name__ == '__main__':
    dummy_input = np.array([[2.5], [3.5]])
    # output = f1_grad_fd(dummy_input)
    # print(output)
    # print(f1_grad_exact(dummy_input))
    dummy_input = np.array([[2.5], [3.5]])
    output = f3_grad_exact(dummy_input)
    # print(output)
    target_output = np.array([[0.02703108, 0.03783601]])
    # print(output)
    # B = np.array([[4, -2], [-2, 4]])
    # a = np.array([[0], [1]])
    # b = np.array([[-2], [1]])

    # B = np.array([[1, 0], [1, 0]])
    # a = np.array([[0], [0]])
    # b = np.array([[0], [0]])
    # check = f1_check_minimum(B, a, b)
