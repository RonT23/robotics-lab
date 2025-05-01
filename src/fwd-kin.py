import numpy as np
import sympy as sp

# symbolic joint variables
q1, q2, q4 = sp.symbols('q1 q2 q4')
#q3, q5, q6, q7 = sp.symbols('q3 q5 q6 q7')

l1, l2, l3, l4, l5 = sp.symbols('l1 l2 l3 l4 l5')
th1, th2 = sp.symbols('th1 th2')

# fixed joint values
q3 = 0.0
q5 = 0.0
q6 = 0.75
q7 = 0.0

# transformation functions
def Rot_x(a):
    return sp.Matrix([
        [1, 0,          0,           0],
        [0, sp.cos(a), -sp.sin(a),  0],
        [0, sp.sin(a),  sp.cos(a),  0],
        [0, 0,          0,           1]
    ])

def Rot_z(a):
    return sp.Matrix([
        [sp.cos(a), -sp.sin(a), 0, 0],
        [sp.sin(a),  sp.cos(a), 0, 0],
        [0,          0,         1, 0],
        [0,          0,         0, 1]
    ])

def Tra_x(d):
    return sp.Matrix([
        [1, 0, 0, d],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def Tra_z(d):
    return sp.Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, d],
        [0, 0, 0, 1]
    ])

# homogeneous transformations
A01 = Rot_z(q1)         @ Tra_z(l1)
A12 = Rot_x(-sp.pi/2)   @ Rot_z(q2)
A23 = Rot_x(sp.pi/2)    @ Rot_z(q3)                 @ Tra_z(l2)
A34 = Rot_x(sp.pi/2)    @ Tra_x(l3)                 @ Rot_z(q4)
A45 = Rot_x(sp.pi/2)    @ Tra_x(l4  * sp.sin(th1))  @ Rot_z(q5) @ Tra_z(l4 * sp.cos(th1))
A56 = Rot_x(sp.pi/2)    @ Rot_z(q6)
A67 = Rot_x(-sp.pi/2)   @ Tra_x(l5  * sp.sin(th2))  @ Rot_z(q7) @ Tra_z(l5 * sp.cos(th2))

sp.pprint(A01, use_unicode=True)
sp.pprint(A12, use_unicode=True)
sp.pprint(A23, use_unicode=True)
sp.pprint(A34, use_unicode=True)
sp.pprint(A45, use_unicode=True)
sp.pprint(A56, use_unicode=True)
sp.pprint(A67, use_unicode=True)

# full transformation
T = sp.simplify(A01 * A12 * A23 * A34 * A45 * A56 * A67)
T = T.evalf(chop=True)  # chop small numbers to zero

sp.pprint(T[0:3, 3], use_unicode=True)

