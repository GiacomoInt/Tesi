"""Implementation of optimization procedures.

This module module contains the implementations of the optimization processes
behind fuzzy inference.

Once loaded, the module preliminarily verifies that some libraries are
installed (notably, Gurobi and TensorFlow), emitting a warning otherwise.
Note that at least one of these libraries is needed in order to solve the
optimization problems involved in the fuzzy inference process.

The module also checks the availability of tqdm, which is used in order to
graphically depict the progress of some learning processes using a progress
bar. However, this package is not strictly needed: if it is not installed,
the above mentioned progress bars will not be displayed.
"""
from random import random

from cvxopt.modeling import variable, constraint
from cvxopt import solvers, matrix

import cvxpy as cp


import numpy as np
import itertools as it
from collections.abc import Iterable
import logging

import mulearn.kernel as kernel

logger = logging.getLogger(__name__)

try:
    from gurobipy import LinExpr, GRB, Model, Env, QuadExpr, GurobiError

    gurobi_ok = True
except ModuleNotFoundError:
    logger.warning('gurobi not available')
    gurobi_ok = False

try:
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    import tensorflow as tf

    tensorflow_ok = True
    from tensorflow.keras.optimizers import Adam

    logging.getLogger('tensorflow').setLevel(logging.ERROR)
except ModuleNotFoundError:
    logger.warning('tensorflow not available')
    tensorflow_ok = False

try:
    import tqdm

    tqdm_ok = True
except ModuleNotFoundError:
    logger.warning('tqdm not available')
    tqdm_ok = False


class Solver:
    """Abstract solver for optimization problems.

    The base class for solvers is :class:`Solver`: it exposes a method
    `solve` which delegates the numerical optimization process to an abstract
    method `solve_problem` and subsequently clips the results to the boundaries
    of the feasible region.
    """

    def solve_problem(self, *args):
        pass

    def solve(self, xs, mus, c, k):
        """Solve optimization phase.

        Build and solve the constrained optimization problem on the basis
        of the fuzzy learning procedure.

        :param xs: Objects in training set.
        :type xs: iterable
        :param mus: Membership values for the objects in `xs`.
        :type mus: iterable
        :param c: constant managing the trade-off in joint radius/error
          optimization.
        :type c: float
        :param k: Kernel function to be used.
        :type k: :class:`mulearn.kernel.Kernel`
        :raises: ValueError if c is non-positive or if xs and mus have
          different lengths.
        :returns: `list` -- optimal values for the independent variables
          of the problem."""
        if c <= 0:
            raise ValueError('c should be positive')

        mus = np.array(mus)
        chis = self.solve_problem(xs, mus, c, k)

        chis_opt = [np.clip(ch, l, u)
                    for ch, l, u in zip(chis, -c * (1 - mus), c * mus)] # noqa

        return chis_opt


class GurobiSolver(Solver):
    """Solver based on gurobi.

    Using this class requires that gurobi is installed and activated
    with a software key. The library is available at no cost for academic
    purposes (see
    https://www.gurobi.com/downloads/end-user-license-agreement-academic/).
    Alongside the library, also its interface to python should be installed,
    via the gurobipy package.
    """

    default_values = {"time_limit": 10 * 60,
                      "adjustment": 0,
                      "initial_values": None}

    def __init__(self, time_limit=default_values['time_limit'],
                 adjustment=default_values['adjustment'],
                 initial_values=default_values['initial_values']):
        """
        Build an object of type GurobiSolver.

        :param time_limit: Maximum time (in seconds) before stopping iterative
          optimization, defaults to 10*60.
        :type time_limit: int
        :param adjustment: Adjustment value to be used with non-PSD matrices,
          defaults to 0. Specifying `'auto'` instead than a numeric value
          will automatically trigger the optimal adjustment if needed.
        :type adjustment: float or `'auto'`
        :param initial_values: Initial values for variables of the optimization
          problem, defaults to None.
        :type initial_values: iterable of floats or None
        """
        self.time_limit = time_limit
        self.adjustment = adjustment
        self.initial_values = initial_values

    def solve_problem(self, xs, mus, c, k):
        """Optimize via gurobi.

        Build and solve the constrained optimization problem at the basis
        of the fuzzy learning procedure using the gurobi API.

        :param xs: objects in training set.
        :type xs: iterable
        :param mus: membership values for the objects in `xs`.
        :type mus: iterable
        :param c: constant managing the trade-off in joint radius/error
          optimization.
        :type c: float
        :param k: kernel function to be used.
        :type k: :class:`mulearn.kernel.Kernel`
        :raises: ValueError if optimization fails or if gurobi is not installed
        :returns: list -- optimal values for the independent variables of the
          problem.
        """
        if not gurobi_ok:
            raise ValueError('gurobi not available')

        m = len(xs)

        with Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with Model('mulearn', env=env) as model:
                model.setParam('OutputFlag', 0)
                model.setParam('TimeLimit', self.time_limit)

                for i in range(m):
                    if c < np.inf:
                        model.addVar(name=f'chi_{i}',
                                     lb=-c * (1 - mus[i]), ub=c * mus[i],
                                     vtype=GRB.CONTINUOUS)
                    else:
                        model.addVar(name=f'chi_{i}', vtype=GRB.CONTINUOUS)

                model.update()
                chis = model.getVars()

                if self.initial_values is not None:
                    for c, i in zip(chis, self.initial_values):
                        c.start = i

                obj = QuadExpr()

                for i, j in it.product(range(m), range(m)):
                    obj.add(chis[i] * chis[j], k.compute(xs[i], xs[j]))

                for i in range(m):
                    obj.add(-1 * chis[i] * k.compute(xs[i], xs[i]))

                if self.adjustment and self.adjustment != 'auto':
                    for i in range(m):
                        obj.add(self.adjustment * chis[i] * chis[i])

                model.setObjective(obj, GRB.MINIMIZE)

                constEqual = LinExpr()
                constEqual.add(sum(chis), 1.0)

                model.addConstr(constEqual, GRB.EQUAL, 1)

                try:
                    model.optimize()
                except GurobiError as e:
                    print(e.message)
                    if self.adjustment == 'auto':
                        s = e.message
                        a = float(s[s.find(' of ') + 4:s.find(' would')])
                        logger.warning('non-diagonal Gram matrix, '
                                       f'retrying with adjustment {a}')
                        for i in range(m):
                            obj.add(a * chis[i] * chis[i])
                        model.setObjective(obj, GRB.MINIMIZE)

                        model.optimize()
                    else:
                        raise e

                if model.Status != GRB.OPTIMAL:
                    raise ValueError('optimal solution not found!')

                print(f"Objective value GUROBI: {str(model.getObjective().getValue())}")


                return [ch.x for ch in chis]

    def __repr__(self):
        obj_repr = f"GurobiSolver("

        for a in ('time_limit', 'adjustment', 'initial_values'):
            if self.__getattribute__(a) != self.default_values[a]:
                obj_repr += f", {a}={self.default_values[a]}"
        return obj_repr + ")"


class TensorFlowSolver(Solver):
    """Solver based on TensorFlow.

    Using this class requires that TensorFlow 2.X is installed."""

    default_values = {"initial_values": "random",
                      "init_bound": 0.1,
                      "n_iter": 100,
                      "optimizer": Adam(learning_rate=1e-4)
                                   if tensorflow_ok else None,  # noqa
                      "tracker": tqdm.trange if tqdm_ok else range,
                      "penalization": 10}

    def __init__(self,
                 initial_values=default_values["initial_values"],
                 init_bound=default_values["init_bound"],
                 n_iter=default_values["n_iter"],
                 optimizer=default_values["optimizer"],
                 tracker=default_values["tracker"],
                 penalization=default_values["penalization"]):
        """Build an object of type TensorFlowSolver.

        :param initial_values: values to be used for initializing the
          independent  variables, either randomly (if set to `'random'`)` or
          seting them to a given sequence of initial values (if set to an
          iterable of floats), defaults to `'random'`.
        :type initial_values: `str` or iterable of `float`
        :param init_bound: Absolute value of the extremes of the interval used
          for random initialization of independent variables, defaults to 0.1.
        :type init_bound: `float`
        :param n_iter: Number of iterations of the optimization process,
          defaults to 100.
        :type n_iter: `int`
        :param optimizer: Optimization algorithm to be used, defaults to Adam
          with learning rate=1e-4 if tensorflow is available, to None otherwise.
        :type optimizer: :class:`tf.keras.optimizers.Optimizer`
        :param tracker: Tool to graphically depict the optimization progress,
          defaults to `tqdm.trange` if tqdm is available, to `range` (i.e., no
          graphical tool) otherwise.
        :type tracker: `object`
        :param penalization: Lagrange penalization for the equality constraint
          in the original problem, defaults to 10.
        :type penalization: `float`
        """
        self.init_bound = init_bound
        self.initial_values = initial_values
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.tracker = tracker
        self.penalization = penalization

    def solve_problem(self, xs, mus, c, k):
        """Optimize via TensorFlow.

        Build and solve the constrained optimization problem on the basis
        of the fuzzy learning procedure using the TensorFlow API.

        :param xs: objects in training set.
        :type xs: iterable
        :param mus: membership values for the objects in `xs`.
        :type mus: iterable
        :param c: constant managing the trade-off in joint radius/error
          optimization.
        :type c: float
        :param k: kernel function to be used.
        :type k: :class:`mulearn.kernel.Kernel`
        :raises: ValueError if optimization fails or if TensorFlow is not
          installed
        :returns: list -- optimal values for the independent variables of the
          problem.
        """ 
        if not tensorflow_ok:
            raise ValueError('tensorflow not available')

        m = len(xs)

        if self.initial_values == 'random':
            chis = [tf.Variable(ch, name=f'chi_{i}',
                                trainable=True, dtype=tf.float32)
                    for i, ch in enumerate(np.random.uniform(-self.init_bound,
                                                             self.init_bound,
                                                             m))]
        elif isinstance(self.initial_values, Iterable):
            chis = [tf.Variable(ch, name=f'chi_{i}',
                                trainable=True, dtype=tf.float32)
                    for i, ch in enumerate(self.initial_values)]
        else:
            raise ValueError("`initial_values` should either be set to "
                             "'random' or to a list of initial values.")

        if type(k) is kernel.PrecomputedKernel:
            gram = k.kernel_computations
        else:
            gram = np.array([[k.compute(x1, x2) for x1 in xs] for x2 in xs])

        def obj():
            kernels = tf.constant(gram, dtype='float32')

            v = tf.tensordot(tf.linalg.matvec(kernels, chis), chis, axes=1)
            v -= tf.tensordot(chis,
                              [k.compute(x_i, x_i) for x_i in xs], axes=1)

            v += self.penalization * tf.math.maximum(0, 1 - sum(chis))
            v += self.penalization * tf.math.maximum(0, sum(chis) - 1)

            if c < np.inf:
                for chi, mu in zip(chis, mus):
                    v += self.penalization * tf.math.maximum(0, chi - c * mu)
                    v += self.penalization *\
                         tf.math.maximum(0, c * (1 - mu) - chi) # noqa

            return v

        for _ in self.tracker(self.n_iter):
            self.optimizer.minimize(obj, var_list=chis)

        return [ch.numpy() for ch in chis]

    def __repr__(self):
        obj_repr = f"TensorFlowSolver("

        for a in ("initial_values", "init_bound", "n_iter",
                  "optimizer", "tracker", "penalization"):
            if self.__getattribute__(a) != self.default_values[a]:
                obj_repr += f", {a}={self.default_values[a]}"
        return obj_repr + ")"


class CVXPYSolver(Solver):

    default_values = {"initial_values": None}

    def __init__(self, initial_values = default_values['initial_values']):

        self.initial_values = initial_values
    
    def solve_problem(self, xs, mus, c, k):
        
        m = len(xs)

        #matrice P
        P = []        
        for i in range(m):
            line = []
            for j in range(m):
                line.append(k.compute(xs[i],xs[j]))
            P.append(line)
   
        P = np.array(P)

        # matrice q
        q = []
        for i in range(m):
            q.append(k.compute(xs[i], xs[i]))
        
        q = np.array(q)

        #matrice G1
        G1 = np.identity(m)
        
        #matrice G2
        G2 = np.identity(m)

        #matrice h1
        h1=[]
        for i in range(m):
            h1.append([mus[i]*c])
        
        h1 = np.array(h1)

        #matrice h2
        h2=[]
        for i in range(m):
            h2.append([(1-mus[i])*c])
        
        h2 = np.array(h2)

        # #matrice h
        # h=[]
        # for i in range(m):
        #     h.append([mus[i]*c])
        #     h.append([(1-mus[i])*c]) 
        
        # h = np.array(h)
        # h = h.astype(np.double)
        
        #matrice A
        A = np.ones(m)

        #matrice b
        b = [1.]

        b = np.array(b)


        #variable
        chis = cp.Variable(shape = (m,1))

        #constraints
        constraints = [cp.sum(chis) == 1]
       
        for i in range(m):            
            constraints.append(mus[i]*c - chis[i] >= 0)  
            constraints.append((1-mus[i])*c + chis[i] >= 0)
      

        expression = cp.Minimize(cp.quad_form(chis, P) - q.T @ chis)

        prob = cp.Problem(expression, constraints)
        prob.solve()

        #[G1 @ chis <= h1, G2 @ chis >= h2, A @ chis == b]

        print(f"Objective value CVXPY: {str(prob.value)}")
        
        chis_value_1 = chis.value[1]
        chis_value =  chis.value    
        chis_T=(chis.value).T

        return np.squeeze(np.array(chis.value))


class CVXOPTSolver(Solver):

    default_values = {"initial_values": None}

    def __init__(self, initial_values = default_values['initial_values']):

        self.initial_values = initial_values
    
    def solve_problem(self, xs, mus, c, k):
        
        m = len(xs)

        #matrice P
        P = []        
        for i in range(m):
            line = []
            for j in range(m):
                line.append(2*(k.compute(xs[i],xs[j])))
            P.append(line)

           
        P = np.array(P)
        P = P.astype(np.double)
        P = matrix(P, tc= 'd')

        # matrice q
        q = []
        for i in range(m):
            q.append(k.compute(xs[i], xs[i]))
        
        q = np.array(q)
        q = q.astype(np.double)
        q = matrix(q, tc= 'd')

        #matrice G
        G=[]
        for i in range(m):
            riga1 = []
            riga2 = []
            for j in range(m):
                if i == j:
                    riga1.append(1.)
                    riga2.append(-1.)
                else:
                    riga1.append(0.)
                    riga2.append(0.)
            G.append(riga1)  
            G.append(riga2)

        G = np.array(G)
        G = G.astype(np.double)
        G = matrix(G, tc= 'd')
        
        #matrice h
        h=[]
        for i in range(m):
            h.append([mus[i]*c])
            h.append([(1-mus[i])*c]) 
        
        h = np.array(h)
        h = h.astype(np.double)
        h = matrix(h, tc= 'd')
        
        # #matrice A
        A = np.ones(m)
        A = A.astype(np.double)
        A = matrix(A,(1,m), tc= 'd')

        #matrice b
        b = [1.]

        b = np.array(b)
        b = b.astype(np.double)
        b = matrix(b, tc= 'd')

        solvers.options['show_progress'] = False

        sol = solvers.qp(P,q,G,h,A,b)

        return  sol['x']


class TensorFlowOptimizedSolver(Solver):
    """Solver based on TensorFlow.

    Using this class requires that TensorFlow 2.X is installed."""

    default_values = {"initial_values": None}

    def __init__(self,
                 initial_values=default_values["initial_values"],
                ):
        
        self.initial_values = initial_values
    
    def original_obj(self, xs, k, chis):
        
        m = len(xs)
        sum1 = 0
        sum2 = 0

        for i, j in it.product(range(m),range(m)):
            sum1 += chis[i]*k.compute(xs[i], xs[i])
            sum2 += chis[i]*chis[j]*k.compute(xs[i], xs[j])

        sum = sum1 - sum2

        return sum
    
    def lagrangian_obj(self,xs, k, chis, mus, c, lambda1, lambda2, lambda3, lambda4):

        m = len(xs)
        last_lambdas = 0

        for i in range(m):            
            last_lambdas += lambda3[i]*(mus[i]*c - chis[i]) + lambda4[i]*(chis[i]+(1-mus[i])*c)
            

        return self.original_obj(xs, k, chis) - (lambda1*sum(chis) -1) - lambda2*(1-sum(chis)) - (last_lambdas)

    def optimize_lagrangian(self, xs, k, chis, mus, c, lambda1, lambda2, lambda3, lambda4):
        

        chis_tf = [tf.Variable(ch, name=f'chi_{i}',
                            trainable=True, dtype=tf.float32)
                    for i, ch in enumerate(chis)]
        
        f_x = self.lagrangian_obj(xs, k, chis_tf, mus, c, lambda1, lambda2, lambda3, lambda4)

        opt = Adam.minimize(f_x, var_list=chis_tf)

        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     for i in range(100):
        #         sess.run(opt)

        return [ch.numpy() for ch in chis_tf]       

    
    def solve_problem(self, xs, mus, c, k):

        m = len(xs)
        chis = np.empty(shape=m)
        #da inizializzare random (?)
        lambda1 = random() 
        lambda2 = random() 
        lambda3 = np.random.rand(m)
        lambda4 = np.random.rand(m)

        gap = 2 # prova
        epsilon = 0.1 # quanto inizializzare? 

        while (gap > epsilon):
            
            chis = self.optimize_lagrangian(xs, k, chis, mus, c, lambda1, lambda2, lambda3, lambda4)

            lambda1 -= (1- sum(chis))
            lambda2 -= (sum(chis) - 1)

            for i in range(m):
                lambda3[i] -= (c*mus[i] - chis[i])
                lambda4[i] -= (chis[i] + (1-mus[i])*c)

            gap = self.original_obj(xs, k, chis) - self.lagrangian_obj(xs, k, chis, mus, c, lambda1, lambda2, lambda3, lambda4)
            
        return chis  
