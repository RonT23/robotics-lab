#!/usr/bin/env python3

"""
Compute state space kinematic matrices for xArm7 robot arm (5 links, 7 joints)
"""

import numpy as np
import math

from utils import Rot, Tra, cross_product

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
        # homogeneous transformations
        A01 = self.tf_A01(r_joints_array)
        A02 = self.tf_A02(r_joints_array)
        A03 = self.tf_A03(r_joints_array)
        A04 = self.tf_A04(r_joints_array)
        A05 = self.tf_A05(r_joints_array)
        A06 = self.tf_A06(r_joints_array)
        A07 = self.tf_A07(r_joints_array)

        # axes of rotation - z vectors in each frame due to DH
        b0 = np.array([0, 0, 1])
        b1 = A01[:3, 2]
        b2 = A02[:3, 2]
        b3 = A03[:3, 2]
        b4 = A04[:3, 2]
        b5 = A05[:3, 2]
        b6 = A06[:3, 2]

        # position vectors
        P07 = A07[:3,3] 
        P17 = P07 - A01[:3,3]
        P27 = P07 - A02[:3,3]
        P37 = P07 - A03[:3,3]
        P47 = P07 - A04[:3,3]
        P57 = P07 - A05[:3,3]
        P67 = P07 - A06[:3,3]
        
        # linear elements
        JL1 = np.cross(b0, P07)
        JL2 = np.cross(b1, P17)
        JL3 = np.cross(b2, P27)
        JL4 = np.cross(b3, P37)
        JL5 = np.cross(b4, P47)
        JL6 = np.cross(b5, P57)
        JL7 = np.cross(b6, P67)
        
        # angular elements
        JA1 = b0 
        JA2 = b1 
        JA3 = b2 
        JA4 = b3 
        JA5 = b4 
        JA6 = b5 
        JA7 = b6

        J_11 = JL1[0]
        J_12 = JL2[0]
        J_13 = JL3[0]
        J_14 = JL4[0]
        J_15 = JL5[0]
        J_16 = JL6[0]
        J_17 = JL7[0]

        J_21 = JL1[1]
        J_22 = JL2[1]
        J_23 = JL3[1]
        J_24 = JL4[1]
        J_25 = JL5[1]
        J_26 = JL6[1]
        J_27 = JL7[1]

        J_31 = JL1[2]
        J_32 = JL2[2]
        J_33 = JL3[2]
        J_34 = JL4[2]
        J_35 = JL5[2]
        J_36 = JL6[2]
        J_37 = JL7[2]

        J_41 = JA1[0]
        J_42 = JA2[0]
        J_43 = JA3[0]
        J_44 = JA4[0]
        J_45 = JA5[0]
        J_46 = JA6[0]
        J_47 = JA7[0]

        J_51 = JA1[1]
        J_52 = JA2[1]
        J_53 = JA3[1]
        J_54 = JA4[1]
        J_55 = JA5[1]
        J_56 = JA6[1]
        J_57 = JA7[1]

        J_61 = JA1[2]
        J_62 = JA2[2]
        J_63 = JA3[2]
        J_64 = JA4[2]
        J_65 = JA5[2]
        J_66 = JA6[2]
        J_67 = JA7[2]

        J = np.matrix([ [ J_11 , J_12 , J_13 , J_14 , J_15 , J_16 , J_17 ],\
                        [ J_21 , J_22 , J_23 , J_24 , J_25 , J_26 , J_27 ],\
                        [ J_31 , J_32 , J_33 , J_34 , J_35 , J_36 , J_37 ],\
                        [ J_41 , J_42 , J_43 , J_44 , J_45 , J_46 , J_47 ],\
                        [ J_51 , J_52 , J_53 , J_54 , J_55 , J_56 , J_57 ],\
                        [ J_61 , J_62 , J_63 , J_64 , J_65 , J_66 , J_67 ]])
        
        return self.jacobian(r_joints_array)

    # Homogeneous transforms
    def tf_A01(self, r_joints_array):
        q1 = r_joints_array[0]
        tf =  Rot("z", q1) @ Tra("z", self.l1)
        return tf

    def tf_A02(self, r_joints_array):
        q2 = r_joints_array[1]
        tf_A12 = Rot("x", -np.pi/2) @ Rot("z", q2)
        tf = np.dot( self.tf_A01(r_joints_array), tf_A12 )
        return tf

    def tf_A03(self, r_joints_array):
        q3 = r_joints_array[2]
        tf_A23 = Rot("x", np.pi/2) @ Rot("z", q3) @ Tra("z", self.l2)
        tf = np.dot( self.tf_A02(r_joints_array), tf_A23 )
        return tf

    def tf_A04(self, r_joints_array):
        q4 = r_joints_array[3]
        tf_A34 = Rot("x", np.pi/2) @ Tra("x", self.l3) @ Rot("z", q4)
        tf = np.dot( self.tf_A03(r_joints_array), tf_A34 )
        return tf

    def tf_A05(self, r_joints_array):
        q5 = r_joints_array[4]
        tf_A45 = Rot("x", np.pi/2) @ Tra("x", self.l4 * np.sin(self.theta1)) @ Rot("z", q5) @ Tra("z", self.l4 * np.cos(self.theta1)) 
        tf = np.dot( self.tf_A04(r_joints_array), tf_A45 )
        return tf

    def tf_A06(self, r_joints_array):
        q6 = r_joints_array[5]
        tf_A56 = Rot("x", np.pi/2) @ Rot("z", q6)     
        tf = np.dot( self.tf_A05(r_joints_array), tf_A56 )
        return tf

    def tf_A07(self, r_joints_array):
        q7 = r_joints_array[6]
        tf_A67 = Rot("x", -np.pi/2) @ Tra("x", self.l5 * np.sin(self.theta2)) @ Rot("z", q7) @ Tra("z", self.l5 * np.cos(self.theta2))      
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
    
    def pose_from_tf(self, tf):
        """
        Extracts position and orientation in Euler angles from a transformation matrix 
        """
        position = tf[:3, 3]  
        orientation = self.rotationMatrixToEulerAngles(tf[:3, :3])
        return np.concatenate((position, orientation))

    def numerical_jacobian(self, q, delta=1e-6):
        J_num = np.zeros((6, 7))
        f_q = self.pose_from_tf(self.tf_A07(q)) 
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = delta
            f_q_perturbed = self.pose_from_tf(self.tf_A07(q + dq))
            J_num[:, i] = (f_q_perturbed - f_q) / delta
        return J_num
    

    def partial_tf_A01(self):
        # Joint 1: alpha=0, a=0
        return np.eye(4)  # Identity, since no Rx or Tx before Rz(theta1)

    def partial_tf_A12(self):
        # Joint 2: Rx(-pi/2), no Tx
        return Rot("x", -np.pi/2)

    def partial_tf_A23(self):
        # Joint 3: Rx(pi/2), no Tx
        return Rot("x", np.pi/2)

    def partial_tf_A34(self):
        # Joint 4: Rx(pi/2), Tx(l3)
        return Rot("x", np.pi/2) @ Tra("x", self.l3)

    def partial_tf_A45(self):
        # Joint 5: Rx(pi/2), Tx(l4 * sin(theta1))
        return Rot("x", np.pi/2) @ Tra("x", self.l4 * np.sin(self.theta1))

    def partial_tf_A56(self):
        # Joint 6: Rx(pi/2), no Tx
        return Rot("x", np.pi/2)

    def partial_tf_A67(self):
        # Joint 7: Rx(-pi/2), Tx(l5 * sin(theta2))
        return Rot("x", -np.pi/2) @ Tra("x", self.l5 * np.sin(self.theta2))
    
    def partial_transform_to(self, joint_index):
        # joint_index: integer from 1 to 7
        
        partial_funcs = [self.partial_tf_A01, self.partial_tf_A12, self.partial_tf_A23,
                        self.partial_tf_A34, self.partial_tf_A45, self.partial_tf_A56,
                        self.partial_tf_A67]
        
        T = np.eye(4)
        for i in range(joint_index-1):  # up to i-1 partial transforms
            T = T @ partial_funcs[i]()
        return T
    
    def joint_axis(self, joint_index):
        T_partial = self.partial_transform_to(joint_index)
        # The z-axis is the 3rd column of the rotation matrix part (top-left 3x3)
        z = T_partial[0:3, 2]
        return z / np.linalg.norm(z)
    
    def joint_origin(self, r_joints_array, joint_index):
        tf_full_funcs = [self.tf_A01, self.tf_A02, self.tf_A03, self.tf_A04,
                        self.tf_A05, self.tf_A06, self.tf_A07]
        T_full = tf_full_funcs[joint_index-1](r_joints_array)
        p = T_full[0:3, 3]
        return p
    
    def jacobian_column(self, r_joints_array, joint_index, p_e):
        z = self.joint_axis(joint_index)
        p = self.joint_origin(r_joints_array, joint_index)
        J_v = np.cross(z, (p_e - p))
        J_w = z
        return J_v, J_w
    
    def jacobian(self, r_joints_array):
        p_e = self.tf_A07(r_joints_array)[0:3, 3]  # end-effector position
        
        J_v = []
        J_w = []
        for i in range(1, 8):  # joints 1 to 7
            jv, jw = self.jacobian_column(r_joints_array, i, p_e)
            J_v.append(jv)
            J_w.append(jw)
        
        J_v = np.column_stack(J_v)
        J_w = np.column_stack(J_w)
        J = np.vstack((J_v, J_w))
        return J