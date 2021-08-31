#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as scs # sparse matrix construction 
import scipy.linalg as scl # linear algebra algorithms
import scipy.optimize as sco # for minimization use
import matplotlib.pylab as plt # for visualization
from scipy.optimize import linprog
from scipy.linalg import LinAlgWarning

import math
import os


def solver(quiz):
    
    def fixed_constraints(N=9):
        rowC = np.zeros(N)
        rowC[0] =1
        rowR = np.zeros(N)
        rowR[0] =1
        row = scl.toeplitz(rowC, rowR)
        ROW = np.kron(row, np.kron(np.ones((1,N)), np.eye(N)))

        colR = np.kron(np.ones((1,N)), rowC)
        col  = scl.toeplitz(rowC, colR)
        COL  = np.kron(col, np.eye(N))

        M = int(np.sqrt(N))
        boxC = np.zeros(M)
        boxC[0]=1
        boxR = np.kron(np.ones((1, M)), boxC) 
        box = scl.toeplitz(boxC, boxR)
        box = np.kron(np.eye(M), box)
        BOX = np.kron(box, np.block([np.eye(N), np.eye(N) ,np.eye(N)]))

        cell = np.eye(N**2)
        CELL = np.kron(cell, np.ones((1,N)))

        return scs.csr_matrix(np.block([[ROW],[COL],[BOX],[CELL]]))


    def clue_constraint(input_quiz, N=9):
        m = np.reshape([int(c) for c in input_quiz], (N,N))
        r, c = np.where(m.T)
        v = np.array([m[c[d],r[d]] for d in range(len(r))])

        table = N * c + r
        table = np.block([[table],[v-1]])

        # it is faster to use lil_matrix when changing the sparse structure.
        CLUE = scs.lil_matrix((len(table.T), N**3))
        for i in range(len(table.T)):
            CLUE[i,table[0,i]*N + table[1,i]] = 1
        # change back to csr_matrix.
        CLUE = CLUE.tocsr() 

        return CLUE

    A0 = fixed_constraints()
    A1 = clue_constraint(quiz)
    A = scs.vstack((A0,A1)).toarray()
    B = np.ones((np.size(A, 0)))
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    K = np.sum(s > 1e-12)
    S_ = np.block([np.diag(s[:K]), np.zeros((K, A.shape[0]-K))])
    A = S_@vh
    B = u.T@B
    B = B[:K]
    c = np.block([ np.ones(A.shape[1]), np.ones(A.shape[1]) ])
    G = np.block([[-np.eye(A.shape[1]), np.zeros((A.shape[1], A.shape[1]))],                   [np.zeros((A.shape[1], A.shape[1])), -np.eye(A.shape[1])]])
    h = np.zeros(A.shape[1]*2)
    H = np.block([A, -A])
    b = B
    res = linprog(c, G, h, H, b, options = {'cholesky': False, 'sym_pos':False})
    X = np.array(res.x)
    x = X[:A.shape[1]] - X[A.shape[1]:]
        # map to board
    z = np.reshape(x, (81, 9))
    
    return (np.reshape(np.array([np.argmax(d)+1 for d in z]), (9,9)))
    

