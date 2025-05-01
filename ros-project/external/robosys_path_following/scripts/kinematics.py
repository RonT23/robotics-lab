#!/usr/bin/env python3

"""
Compute state space kinematic matrices for xArm7 robot arm (5 links, 7 joints)
"""

import numpy as np
import math

def interpolate_points(P0, P1, n):
    """
    This function is used to interpolate two points in space using linear interpolation 
    @param P0 : the initial position in (X, Y, Z)
    @param P1 : the final position in (X, Y, Z)
    @param n  : the number of interpolated values to produce
    @return   : returns the set of interpolated positions
    """
    P0 = np.array(P0, dtype=float)
    P1 = np.array(P1, dtype=float)

    linear_points    = []

    # main interpolation loop
    for t in np.linspace(0, 1, n + 2):
        linear_points.append(P0 + (P1 - P0) * t )

    return np.array(linear_points)

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

class xArm7_kinematics():
    def __init__(self):

        self.l1 = 0.267
        self.l2 = 0.293
        self.l3 = 0.0525
        self.l4 = 0.3512
        self.l5 = 0.1232

        self.theta1 = 0.2225 #(rad) (=12.75deg)
        self.theta2 = 0.6646 #(rad) (=38.08deg)

    # IK
    def compute_angles(self, ee_position):
        """
        This function computes the configuration of joints for the robot arm through 
        inverse kinematics. It controls joints q1, q2 and q4.
        """

        # constants
        k1 = self.l4 * np.cos(self.theta1) + self.l5 * np.sin(self.theta2 + math.atan(0.73 / 0.681))
        k2 = self.l4 * np.sin(self.theta1) + self.l5 * np.sin(self.theta2 + math.atan(-0.681 / 0.73))
        
        # get the input position
        P_ex, P_ey, P_ez = ee_position

        # compute joint 1 position
        q1 = math.atan2(P_ey, P_ex)   
        
        # prevent division by zero
        if abs(abs(q1) - np.pi/2) < 1e-3:
            q1 += (0.001 if q1 > 0 else -0.001)

        X = P_ex / np.cos(q1)
        Z = P_ez - self.l1
        L = ( X**2 + Z**2 - self.l2**2 - self.l3**2 - k1**2 - k2**2 ) / 2
 
        t1 = self.l2 * k2 + self.l3 * k1
        t2 = self.l3 * k2 - self.l2 * k1
 
        R   = np.sqrt(t1**2 + t2**2)
        phi = math.atan2(t2, t1)
 
        sin_q4_phi = L/R
        cos_q4_phi = np.sqrt(1 - sin_q4_phi**2)
        q4 = math.atan2(sin_q4_phi, cos_q4_phi) - phi
       
#        q4 = math.asin(L/R) - phi

        u1 = self.l2 - k1 * np.cos(q4) + k2 * np.sin(q4)
        u2 = self.l3 + k1 * np.sin(q4) + k2 * np.cos(q4)

        det = u1**2 + u2**2
        sin_q2 = (u1 * X - u2 * Z) / det
        cos_q2 = (u2 * X + u1 * Z) / det
        q2 = math.atan2(sin_q2, cos_q2)

        joint_1 = q1
        joint_2 = q2
        joint_3 = 0.0
        joint_4 = q4
        joint_5 = 0.0
        joint_6 = 0.75
        joint_7 = 0.0

        joint_angles = np.array([ joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7 ], dtype=float)

        return joint_angles

    # DK
    def compute_jacobian(self, r_joints_array):

        J_11 = 0
        J_12 = 0
        J_13 = 0
        J_14 = 0
        J_15 = 0
        J_16 = 0
        J_17 = 0

        J_21 = 0
        J_22 = 0
        J_23 = 0
        J_24 = 0
        J_25 = 0
        J_26 = 0
        J_27 = 0

        J_31 = 0
        J_32 = 0
        J_33 = 0
        J_34 = 0
        J_35 = 0
        J_36 = 0
        J_37 = 0

        J_41 = 0
        J_42 = 0
        J_43 = 0
        J_44 = 0
        J_45 = 0
        J_46 = 0
        J_47 = 0

        J_51 = 0
        J_52 = 0
        J_53 = 0
        J_54 = 0
        J_55 = 0
        J_56 = 0
        J_57 = 0

        J_61 = 0
        J_62 = 0
        J_63 = 0
        J_64 = 0
        J_65 = 0
        J_66 = 0
        J_67 = 0

        J = np.matrix([ [ J_11 , J_12 , J_13 , J_14 , J_15 , J_16 , J_17 ],\
                        [ J_21 , J_22 , J_23 , J_24 , J_25 , J_26 , J_27 ],\
                        [ J_31 , J_32 , J_33 , J_34 , J_35 , J_36 , J_37 ],\
                        [ J_41 , J_42 , J_43 , J_44 , J_45 , J_46 , J_47 ],\
                        [ J_51 , J_52 , J_53 , J_54 , J_55 , J_56 , J_57 ],\
                        [ J_61 , J_62 , J_63 , J_64 , J_65 , J_66 , J_67 ]])
        return J

    # Foundamental transformation operations
    def Rot(self, axis, angle):
        """
        Construct a 4x4 rotation matrix about a given axis.
        """
        R = np.eye(4)
        c = np.cos(angle)
        s = np.sin(angle)

        if axis == "x":
            R[1, 1] = c;  R[1, 2] = -s
            R[2, 1] = s;  R[2, 2] = c
        elif axis == "y":
            R[0, 0] = c;  R[0, 2] = s
            R[2, 0] = -s; R[2, 2] = c
        elif axis == "z":
            R[0, 0] = c;  R[0, 1] = -s
            R[1, 0] = s;  R[1, 1] = c
        else:
            print("[ERROR] Rot : Invalid axis. Returning identity.")
        return R

    def Tra(self, axis, displacement):
        """
        Construct a 4x4 translation matrix along a given axis.
        """
        T = np.eye(4)
        if axis == "x":
            T[0, 3] = displacement
        elif axis == "y":
            T[1, 3] = displacement
        elif axis == "z":
            T[2, 3] = displacement
        else:
            print("[ERROR] Tra : Invalid axis. Returning identity.")
        return T
    
    # Homogeneous transforms
    def tf_A01(self, r_joints_array):
        q1 = r_joints_array[0]
        tf =  self.Rot("z", q1) @ self.Tra("z", self.l1)
        return tf

    def tf_A02(self, r_joints_array):
        
        q2 = r_joints_array[1]
        tf_A12 = self.Rot("x", -np.pi/2) @ self.Rot("z", q2)
        tf = np.dot( self.tf_A01(r_joints_array), tf_A12 )
        return tf

    def tf_A03(self, r_joints_array):

        q3 = r_joints_array[2]
        tf_A23 = self.Rot("x", np.pi/2) @ self.Rot("z", q3) @ self.Tra("z", self.l2)
        tf = np.dot( self.tf_A02(r_joints_array), tf_A23 )
        return tf

    def tf_A04(self, r_joints_array):

        q4 = r_joints_array[3]
        tf_A34 = self.Rot("x", np.pi/2) @ self.Tra("x", self.l3) @ self.Rot("z", q4)
        tf = np.dot( self.tf_A03(r_joints_array), tf_A34 )
        return tf

    def tf_A05(self, r_joints_array):

        q5 = r_joints_array[4]
        tf_A45 = self.Rot("x", np.pi/2) @ self.Tra("x", self.l4 * np.sin(self.theta1)) @ self.Rot("z", q5) @ self.Tra("z", self.l4 * np.cos(self.theta1)) 
        tf = np.dot( self.tf_A04(r_joints_array), tf_A45 )
        return tf

    def tf_A06(self, r_joints_array):

        q6 = r_joints_array[5]
        tf_A56 = self.Rot("x", np.pi/2) @ self.Rot("z", q6)     
        tf = np.dot( self.tf_A05(r_joints_array), tf_A56 )
        return tf

    def tf_A07(self, r_joints_array):

        q7 = r_joints_array[6]
        tf_A67 = self.Rot("x", -np.pi/2) @ self.Tra("x", self.l5 * np.sin(self.theta2)) @ self.Rot("z", q7) @ self.Tra("z", self.l5 * np.cos(self.theta2))      
        tf = np.dot( self.tf_A06(r_joints_array), tf_A67 )
        return tf

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self, R) :

        assert(isRotationMatrix(R))

        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])


