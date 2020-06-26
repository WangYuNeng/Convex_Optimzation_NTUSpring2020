import numpy as np
from numpy.linalg import inv
import cvxpy as cp
import matplotlib.pyplot as plt

ALPHA = 0.1
BETA = 0.7
EP_INNER = 1e-5
EP_OUTER = 1e-10
MIU = 20
SETTING = 1

class Optimizer:
    def __init__(self, func):
        self.obj_func = func

    def step(self, x):
        
        newton_step = -inv(self.obj_func.hessian(x)).dot(self.obj_func.grad(x)).reshape((-1,1))
        newton_decr = np.sqrt((-self.obj_func.grad(x).transpose().dot(newton_step)))[0, 0]

        newton_step = newton_step.squeeze()
        s = 1
        while not self.obj_func.is_feasible(x + s * newton_step):
            s *= BETA
        while (self.obj_func.at(x + s * newton_step) > self.obj_func.at(x) - ALPHA * s * newton_decr * newton_decr):
            s *= BETA
        x = x + s * newton_step
        # print(t)
        return (x, newton_decr * newton_decr / 2 < EP_INNER, newton_decr, s)

class Function:
    def __init__(self, A, b, C, d, t):
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        self.t = t
        self.obj_hessian_base = [ np.matmul(A[i][None].transpose(), A[i][None]) for i in range(len(A))]
        self.barrier_hessian_base = [ np.matmul(C[i][None].transpose(), C[i][None]) for i in range(len(C))]
        pass

    def at_obj(self, x):
        ret_val = 0
        for a, b in zip(self.A, self.b):
            ret_val += np.exp(a.dot(x)-b)
        return ret_val

    def at(self, x):
        ret_val = self.t * self.at_obj(x)
        for c, d in zip(self.C, self.d):
            ret_val += -np.log(d-c.dot(x))
        return ret_val

    def grad(self, x):
        ret_val = 0
        for a, b in zip(self.A, self.b):
            ret_val += np.exp(a.dot(x)-b) * a
        ret_val *= self.t
        for c, d in zip(self.C, self.d):
            ret_val += c / (d-c.dot(x))
        return ret_val.reshape((-1,1))

    def hessian(self, x):
        ret_val = 0
        for a, b, h in zip(self.A, self.b, self.obj_hessian_base):
            ret_val += np.exp(a.dot(x)-b) * h
        ret_val *= self.t
        for c, d, h in zip(self.C, self.d, self.barrier_hessian_base):
            ret_val += h / ((d-c.dot(x))*(d-c.dot(x)))
        return ret_val
    
    def is_feasible(self, x):
        for c, d in zip(self.C, self.d):
            if d-c.dot(x) < 0:
                return False
        return True

def my_hw3(x, C, d, A, b, t):
    print("Setting {}".format(SETTING))
    print("A = ", A)
    print("b = ", b)
    print("C = ", C)
    print("d = ", d)
    l = 0 # iter_cnt
    ls = [0]
    xs = []
    f0s, fs, ds, ss, gs = [], [], [], [], [] # f0(x), f(x), newton_decr, backtrack_s, dual_gap
    while True:
        func = Function(A, b, C, d, t)
        optimizer = Optimizer(func)
        converged = False
        xs.append(x)
        while not converged:
            f0s.append(func.at_obj(x)[0])
            fs.append(func.at(x)[0])
            (x, converged, decr, s) = optimizer.step(x)
            ds.append(decr)
            ss.append(s)
            gs.append(len(d) / t)
            l += 1
            # print(x)
        ls.append(l - ls[-1])
        print(
            "Iter {}: x =".format(len(ls)-1), x, "inner step = {}".format(ls[-1])
        )
        if len(d) / t < EP_OUTER:
            break
        t *= MIU
    print("Total number of Newton steps = {}".format(l))
    print("Optimal value = {}".format(func.at_obj(x)[0]))
    print("Optimal x =", x)
    '''
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.plot(xs[0][0], xs[0][1], marker='o', c='purple', label='starting point')
    plt.plot([x[0] for x in xs[1:]], [x[1] for x in xs[1:]], marker='o', c='b', label='central path')
    x2 = np.arange(-0.01, 0.06, 0.001)
    for i, (c, dd) in enumerate(zip(C, d)):
        plt.plot((-c[1]*x2+dd)/c[0], x2, label='constraint{}'.format(i+1))
    plt.legend()
    plt.show()
    plt.clf()

    x_axis = [i for i in range(len(fs))]

    plt.xlabel('iter')
    plt.yscale('log')
    plt.plot(x_axis, gs, marker='o', label='duality gap (MIU={})'.format(MIU))
    plt.legend(loc=1)
    # plt.savefig("figures/{}_{}_n.png".format(SETTING, MIU))
    # plt.clf()

    plt.xlabel('iter')
    plt.yscale('log')
    plt.plot(x_axis, gs, marker='^', label='duality gap')
    plt.plot(x_axis, [val*val/2 for val in ds], marker='o', label='lambda(x)^2/2')
    plt.legend(loc=1)
    plt.savefig("figures/{}_{}_p.png".format(SETTING, MIU))
    plt.clf()

    plt.xlabel('iter')
    plt.yscale('log')
    plt.plot(x_axis, gs, marker='^', label='duality gap')
    plt.plot(x_axis, ss, marker='o', label='backtracking s')
    plt.legend(loc=1)
    plt.savefig("figures/{}_{}_q.png".format(SETTING, MIU))
    plt.clf()

    plt.xlabel('iter')
    plt.yscale('log')
    plt.plot(x_axis, f0s, marker='o', label='f_0(x)')
    plt.legend(loc=1)
    plt.savefig("figures/{}_{}_r.png".format(SETTING, MIU))
    plt.clf()
    '''

    return x
    
def cvx_hw3(A, b, C, d, x):
    x = cp.Variable(len(x), value=x.reshape(-1,))
    prob = cp.Problem(cp.Minimize(cp.sum(cp.exp(A @ x - b.squeeze())))
                        , [C @ x <= d.squeeze()])
    prob.solve()
    print(prob.value)
    # Print result.
    x = x.value

    return x

if __name__ == "__main__":
    A = np.array([[1, 3], [1, -3], [-1, 0]])
    b = np.array([[0.1], [0.1], [0.1]])
    C = np.array([[-1, 0], [-1, -1]])
    d = np.array([[0.3], [0.2]])
    SETTING=1
    my_hw3(np.array([[0], [0]]).reshape(-1,), C, d, A, b, 1)
    # print(cvx_hw3(A, b, C, d, np.array([[0], [0]])))
    
    print("")
    SETTING=2
    d = np.array([[0.3], [0.3]])
    my_hw3(np.array([[0], [0]]).reshape(-1,), C, d, A, b, 1)

    print("")
    SETTING=3
    d = np.array([[0.4], [0.4]])
    my_hw3(np.array([[0], [0]]).reshape(-1,), C, d, A, b, 1)

