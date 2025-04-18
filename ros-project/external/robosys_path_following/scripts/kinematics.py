#!/usr/bin/env python3

"""
Compute state space kinematic matrices for xArm7 robot arm (5 links, 7 joints)
"""

import numpy as np
import math

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

        pass

    def compute_angles(self, ee_position):
        """
        This function computes the configuration of joints for the robot arm through 
        inverse kinematics. It controls joints q1, q2 and q4.
        """
        joint_1 = 0.0
        joint_2 = 0.0
        joint_3 = 0.0
        joint_4 = 0.0
        joint_5 = 0.0
        joint_6 = 0.75
        joint_7 = 0.0

        joint_angles = np.array([ joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7 ], dtype=float)

        return joint_angles

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

    def tf_A01(self, r_joints_array):
        tf = np.matrix([[1 , 0 , 0 , 0],\
                        [0 , 1 , 0 , 0],\
                        [0 , 0 , 1 , 0],\
                        [0 , 0 , 0 , 1]])
        return tf

    def tf_A02(self, r_joints_array):
        tf_A12 = np.matrix([[1 , 0 , 0 , 0],\
                            [0 , 1 , 0 , 0],\
                            [0 , 0 , 1 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A01(r_joints_array), tf_A12 )
        return tf

    def tf_A03(self, r_joints_array):
        tf_A23 = np.matrix([[1 , 0 , 0 , 0],\
                            [0 , 1 , 0 , 0],\
                            [0 , 0 , 1 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A02(r_joints_array), tf_A23 )
        return tf

    def tf_A04(self, r_joints_array):
        tf_A34 = np.matrix([[1 , 0 , 0 , 0],\
                            [0 , 1 , 0 , 0],\
                            [0 , 0 , 1 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A03(r_joints_array), tf_A34 )
        return tf

    def tf_A05(self, r_joints_array):
        tf_A45 = np.matrix([[1 , 0 , 0 , 0],\
                            [0 , 1 , 0 , 0],\
                            [0 , 0 , 1 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A04(r_joints_array), tf_A45 )
        return tf

    def tf_A06(self, r_joints_array):
        tf_A56 = np.matrix([[1 , 0 , 0 , 0],\
                            [0 , 1 , 0 , 0],\
                            [0 , 0 , 1 , 0],\
                            [0 , 0 , 0 , 1]])
        tf = np.dot( self.tf_A05(r_joints_array), tf_A56 )
        return tf

    def tf_A07(self, r_joints_array):
        tf_A67 = np.matrix([[1 , 0 , 0 , 0],\
                            [0 , 1 , 0 , 0],\
                            [0 , 0 , 1 , 0],\
                            [0 , 0 , 0 , 1]])
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
