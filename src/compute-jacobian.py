import numpy as np
import sympy as sp

q1, q2, q4         = sp.symbols('q1 q2 q4')
q3, q5, q6, q7     = sp.symbols('q3 q5 q6 q7')
l1, l2, l3, l4, l5 = sp.symbols('l1 l2 l3 l4 l5')
th1, th2           = sp.symbols('th1 th2')

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

# helper vectors 
r = sp.Matrix([[0],[0],[0],[1]])
z = sp.Matrix([[0], [0], [1]])      # revolute joints around z-axis

# Compute subsequent homogeneous transforms
A02 = A01 @ A12
A03 = A02 @ A23
A04 = A03 @ A34 
A05 = A04 @ A45 
A06 = A05 @ A56
A07 = A06 @ A67 

# directional vectors
b0 = sp.Matrix(z)
b1 = sp.Matrix(A01[0:3,0:3]) @ sp.Matrix(z)
b2 = sp.Matrix(A02[0:3,0:3]) @ sp.Matrix(z)
b3 = sp.Matrix(A03[0:3,0:3]) @ sp.Matrix(z)
b4 = sp.Matrix(A04[0:3,0:3]) @ sp.Matrix(z)
b5 = sp.Matrix(A05[0:3,0:3]) @ sp.Matrix(z)
b6 = sp.Matrix(A06[0:3,0:3]) @ sp.Matrix(z)

P01 = sp.Matrix((A01 @ r)[0:3, :])
P02 = sp.Matrix((A02 @ r)[0:3, :])
P03 = sp.Matrix((A03 @ r)[0:3, :])
P04 = sp.Matrix((A04 @ r)[0:3, :])
P05 = sp.Matrix((A05 @ r)[0:3, :])
P06 = sp.Matrix((A06 @ r)[0:3, :])
P07 = sp.Matrix((A07 @ r)[0:3, :])

# position vectors
P17 = sp.Matrix(P07 - P01)
P27 = sp.Matrix(P07 - P02)
P37 = sp.Matrix(P07 - P03)
P47 = sp.Matrix(P07 - P04)
P57 = sp.Matrix(P07 - P05)
P67 = sp.Matrix(P07 - P06)

# Jacobian vectors elements
JL1, JA1 = b0.cross(P07), b0
JL2, JA2 = b1.cross(P17), b1
JL3, JA3 = b2.cross(P27), b2
JL4, JA4 = b3.cross(P37), b3
JL5, JA5 = b4.cross(P47), b4
JL6, JA6 = b5.cross(P57), b5
JL7, JA7 = b6.cross(P67), b6

# Form the jacobian matrix
J = sp.Matrix.hstack(
    sp.Matrix.vstack(JL1, JA1),
    sp.Matrix.vstack(JL2, JA2),
    sp.Matrix.vstack(JL3, JA3),
    sp.Matrix.vstack(JL4, JA4),
    sp.Matrix.vstack(JL5, JA5),
    sp.Matrix.vstack(JL6, JA6),
    sp.Matrix.vstack(JL7, JA7)
)

sp.pprint(J, use_unicode=True)