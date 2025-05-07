#!/usr/bin/env python3
"""
Engineer    : Ronaldo Tsela
Date        : 2/5/2025
Description : This is a ROS node that implements a 7 DOF position and orientation 
            path planning and control algorithm for the xArm7 cobot manipulator. 
            This is part of the final project on Control and Robotics MSc program 
            at NTUA for the academic year 2024/2025.
Note        : This script is an extended and refined version of the provided one.
"""

import sys
import rospy

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

import numpy as np

import time as t

from kinematics import xArm7_kinematics

# Additional utility functions created
from utils import interpolate_points
from utils import read_task_file

from tf.transformations import quaternion_matrix, quaternion_from_euler, quaternion_multiply, quaternion_inverse
matrix = quaternion_matrix([1, 0, 0, 0])

C_USER_TASK_FILE = "task_file_2.txt"
C_DEBUG_MODE = False

class xArm7_controller():
    """
    Class to compute and publish joints positions
    """
    
    def __init__(self,rate):

         #-------- Robot Manipulator Setup ---------------
        # Initialize the xArm7 kinematics handler
        self.kinematics = xArm7_kinematics()

        # Initial joints' angular positions
        self.joint_angpos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # joints' angular velocities
        self.joint_angvel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Initial joints' states
        self.joint_states = JointState()

        # timer instance
        self.time_prev = 0.0

        #-------- ROS Environment Setup ------------------     
        # Subscribe to the ROS topic where the xArm7 publishes the joint state
        self.joint_states_sub = rospy.Subscriber('/xarm/joint_states', JointState, self.joint_states_callback, queue_size=1)
        
        # Setup topics from where xArm7 listens for joint position angles 
        self.joint_pos_pub = []
        for i in range(0, 7):
            self.joint_pos_pub.append(rospy.Publisher(f'/xarm/joint{i+1}_position_controller/command', Float64, queue_size=1))

        # Set the publishing rate
        self.rate = rate
        self.pub_rate = rospy.Rate(rate)

        self.publish()

    def joint_states_callback(self, msg):
        self.joint_states = msg
    
        # Functions used for evaluation and debugging
    
    def print_current_joint_position(self, extra_msg=None):
        print(f"\nCurrent Joint Positions {extra_msg}")
        joint_positions = self.joint_states.position
        for joint, position in enumerate(joint_positions):
            print(f"Joint_{joint} : {position}")

    def print_target_joint_position(self, position):
        print("\nTarget Joint Positions")
        for joint, position in enumerate(position):
            print(f"Joint_{joint} : {position}")

    def print_current_ee_position(self, extra_msg=None):
        print(f"\nCurrent End Effector Position {extra_msg}")
        joint_positions = self.joint_states.position
        ee_position = self.kinematics.tf_A07(joint_positions)[0:3, 3]
        label = ["x", "y", "z"]

        for i, p in enumerate(ee_position):
            value = np.asscalar(p) if hasattr(np, 'asscalar') else p.item()
            print(f"Pe{label[i]} : {value:.4f}")

    def print_target_ee_position(self, position):
        print("\nTarget End Effector Position")
        label = ["x", "y", "z"]

        for i, p in enumerate(position):
            print(f"Pt{label[i]} : {p:.4f}")

    def print_target_fk_solution(self, position):
        print(f"\nComputed Position from IK solutions")
        ee_position = self.kinematics.tf_A07(position)[0:3, 3]
        label = ["x", "y", "z"]

        for i, p in enumerate(ee_position):
            value = np.asscalar(p) if hasattr(np, 'asscalar') else p.item()
            print(f"Pe{label[i]} : {value:.4f}")
    
    def print_current_ee_orientation(self, extra_msg=None):
        print(f"\nCurrent EE Orientation {extra_msg}")
        A07 = self.kinematics.tf_A07(self.joint_states.position)
        ori = self.kinematics.rotationMatrixToEulerAngles( A07[0:3,0:3] )
        labels= ['x', 'y', 'z']
        for i, r in enumerate(ori):
            print(f" {labels[i]}: {r}")

    ###

    def wrap_angle(self, angle):
        wrapped_angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return wrapped_angle

    def full_control_algorithm(self, P=None, O=None, tolerance=1e-4, a=0.001):
        
        # find the current joint configuration
        self.joint_angpos = self.joint_states.position

        # task loop
        for i in range(0, len(P)-1):

            if not C_DEBUG_MODE:
                print(f"\r PROGRESS: {round(i / len(P) * 100.0, 2)}%", end="", flush=True)

            # Task control Loop
            while True:
                
                if rospy.is_shutdown():
                    return False
                
                # Compute each transformation matrix wrt the base frame from joints' angular positions
                self.A07 = self.kinematics.tf_A07(self.joint_angpos)
                A07_real = self.kinematics.tf_A07(self.joint_states.position)

                # get the current position and orientation of the end effector
                ee_pos = A07_real[0:3,3]
                ee_ori = self.kinematics.rotationMatrixToEulerAngles( A07_real[0:3,0:3] )
                
                # print(f"\n\nEE pos = \n{ee_pos}")
                # print(f"EE ori = \n{ee_ori}")

                # Compute the pseudo inverse jacobian matrix
                J = self.kinematics.compute_jacobian(self.joint_angpos)
                pinvJ = np.linalg.pinv(J)

                # compute the error
                p_err = P[i] - ee_pos
                o_err = O[i] - ee_ori

                err = np.concatenate([p_err, o_err])

                # Compute the joints position
                self.joint_angvel = pinvJ @ err
                self.joint_angvel = np.asarray(self.joint_angvel).flatten()

                self.joint_angpos += self.joint_angvel * a
                self.joint_angpos = [self.wrap_angle(angle) for angle in self.joint_angpos]

                print(err)
            
                if np.linalg.norm(err) <= tolerance:
                    break

                # print(f"\n\nJoint velocities: \n{self.joint_angvel}")
                # print(f"\n\nJoint positions: \n {self.joint_angpos}")
                # print(f"\n\nJoint poisiton calculated: \n{self.joint_angpos}")
                # print(f"\n\nJacobian Matrix = \n{J}")
                # print(f"\nPseudo inverse Jacobian Matrix  = \n{pinvJ}")

                if C_DEBUG_MODE:
                    self.print_target_ee_position(P[i])
                    self.print_current_ee_position(extra_msg="-Before Command")
                    self.print_target_joint_position(self.joint_angpos)
                    self.print_current_joint_position(extra_msg="-Before Command")
                    self.print_target_fk_solution(self.joint_angpos)

                # Publish the new joint's angular positions
                for j in range(7):
                    self.joint_pos_pub[j].publish(self.joint_angpos[j])
                
                # let it execute
                self.pub_rate.sleep()

                if C_DEBUG_MODE:
                    self.print_current_ee_position(extra_msg="-After Command")

        if not C_DEBUG_MODE:
            print("\r PROGRESS: Complete!", end="", flush=True)
        
        return True

    def publish(self):
        """
        Implements the task 2:
        Full position and orientation control.
        """

        # set initial configuration
        self.joint_angpos = [0, 0.75, 0, 1.5, 0, 0.75, 0]
       
        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()
        
        self.joint_pos_pub[3].publish(self.joint_angpos[3])
        tmp_rate.sleep()

        self.joint_pos_pub[1].publish(self.joint_angpos[1])
        self.joint_pos_pub[5].publish(self.joint_angpos[5])
        tmp_rate.sleep()
        ##

        # read the user-specified waypoints file
        P_user = read_task_file(C_USER_TASK_FILE)

        if C_DEBUG_MODE:
            print(f"\nUser-defined waypoints:\n{P_user}")

        while not rospy.is_shutdown():
            for i in range(0, len(P_user)-1):

                P0, O0 = P_user[i][0:3], P_user[i][3:6]
                P1, O1 = P_user[i+1][0:3], P_user[i+1][3:6]

                P = interpolate_points(P0, P1, self.rate)
                O = interpolate_points(O0, O1, self.rate)

                print(f"\b TASK: {P0}, {O0} ==> {P1}, {O1} \n", end="")
                
                self.full_control_algorithm(tolerance=1e-4, P=P, O=O)

                #self.position_control_algorithm(tolerance=1e-4, P=P)
                # A07 = self.kinematics.tf_A07(self.joint_states.position)
                # cur_pos = A07[0:3,3]
                # cur_ori = self.kinematics.rotationMatrixToEulerAngles(A07[0:3,0:3])
                
                # with open("pose_log.txt", "a") as log_file:
                #     log_file.write(f"set_pose: {cur_pos[0]:.2f}, {cur_pos[1]:.2f}, {cur_pos[2]:.2f}, {cur_ori[0]:.2f}, {cur_ori[1]:.2f}, {cur_ori[2]:.2f}\n")

            break 

        print("\nINFO: Task Complete")
        print("INFO: Press Ctrl + C to terminate")

    def turn_off(self):
        rospy.loginfo("Shutting down ROS")
        sys.exit(0)

def controller_py():
    rospy.init_node('controller_node', anonymous=True)
    
    rate = rospy.get_param("/rate")
    controller = xArm7_controller(rate)
    
    rospy.on_shutdown(controller.turn_off)
    rospy.spin()

if __name__ == '__main__':
    try:
        controller_py()
    except rospy.ROSInterruptException:
        pass
