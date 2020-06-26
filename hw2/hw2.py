import numpy as np
from numpy.linalg import inv
import cvxpy as cp
import matplotlib.pyplot as plt

ALPHA = 0.1
BETA = 0.7
EPSILON = 1e-10

class Optimizer:
    def __init__(self, func):
        self.obj_func = func

    def step(self, x):
        
        newton_step = -inv(self.obj_func.hessian(x)).dot(self.obj_func.grad(x)).reshape((2,1))
        newton_decr = np.sqrt((-self.obj_func.grad(x).transpose().dot(newton_step)))

        t = 1
        while self.obj_func.at(x + t * newton_step) > self.obj_func.at(x) - ALPHA * t * newton_decr * newton_decr:
            t *= BETA
        x = x + t * newton_step
        # print(t)
        return (x, newton_decr * newton_decr / 2 < EPSILON, newton_decr[0,0], t)

class Function:
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.hessian_base = [ np.matmul(A[i][None].transpose(), A[i][None]) for i in range(len(A))]
        pass

    def at(self, x):
        ret_val = 0
        for a, b in zip(self.A, self.b):
            ret_val += np.exp(a.dot(x)-b)
        return ret_val

    def grad(self, x):
        ret_val = 0
        for a, b in zip(self.A, self.b):
            ret_val += np.exp(a.dot(x)-b) * a
        return ret_val.reshape((-1,1))

    def hessian(self, x):
        ret_val = 0
        for a, b, h in zip(self.A, self.b, self.hessian_base):
            ret_val += np.exp(a.dot(x)-b) * h
        return ret_val

def my_hw2(A, b, x):
    func = Function(A, b)
    optimizer = Optimizer(func)
    converged = False
    fs, ds, ts = [], [], []
    while not converged:
        fs.append(func.at(x)[0])
        (x, converged, decr, t) = optimizer.step(x)
        ds.append(decr)
        ts.append(t)
        # print(x)
    '''
    print("fs: ",fs)
    print("ds: ",ds)
    print("ts: ",ts)
    
    x_axis = [i for i in range(len(fs))]
    plt.xlabel('iter')
    plt.ylabel('f(x)')
    plt.plot(x_axis, fs, marker='o')
    plt.show()
    plt.ylabel('lambda(x)')
    plt.yscale('log')
    plt.plot(x_axis, ds, marker='o')
    plt.show()
    plt.ylabel('lambda(x)^2/2')
    plt.yscale('log')
    plt.plot(x_axis, [val*val/2 for val in ds], marker='o')
    plt.show()
    plt.ylabel('t')
    plt.plot(x_axis, ts, marker='o')
    plt.show()
    '''
    return x
    
def cvx_hw2(A, b, x):
    x = cp.Variable(len(x), value=x.reshape(-1,))
    prob = cp.Problem(cp.Minimize(cp.sum(cp.exp(A @ x - b.squeeze()))))
    prob.solve()
    # print(prob.value)
    # Print result.
    x = x.value

    return x

if __name__ == "__main__":
    A = np.array([[1, 3], [1, -3], [-1, 0]])
    b = np.array([[0.1], [0.1], [0.1]])
    
    print("ID = b05901025")
    print("### Setting 1: ")
    print("A =\n", A)
    print("b =\n", b)
    print(">>> results from Newtown’s Method")
    print("Opt. of x =", my_hw2(A, b, np.array([[0], [0]])).reshape(-1,))
    print(">>> results from CVX")
    print("Opt. of x =", cvx_hw2(A, b, np.array([[0], [0]])))

    A = np.array([[1, 2], [1, -2], [-1, 0]])
    b = np.array([[0.2], [0.1], [0.3]])
    
    print("### Setting 2: ")
    print("A =\n", A)
    print("b =\n", b)
    print(">>> results from Newtown’s Method")
    print("Opt. of x =", my_hw2(A, b, np.array([[0], [0]])).reshape(-1,))
    print(">>> results from CVX")
    print("Opt. of x =", cvx_hw2(A, b, np.array([[0], [0]])))

