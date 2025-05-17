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
from utils import cubic_interpolation
from utils import linear_interpolation
from utils import read_task_file
#from utils import generate_trajectory_5th_roder

# from tf.transformations import quaternion_matrix, quaternion_from_euler, quaternion_multiply, quaternion_inverse
# matrix = quaternion_matrix([1, 0, 0, 0])

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
    def full_control_with_linear_interpolation(self, P, O):
        
        self.joint_angpos = self.joint_states.position
        X = np.zeros(6)
        kp = 2 * np.eye(6)
        kd = 0.1 * np.eye(6)
        dt = 0.0

        for i in range(0, len(P)):
            if rospy.is_shutdown():
                return False

            if not C_DEBUG_MODE:
               print(f"\r PROGRESS: {round(i / len(P) * 100.0, 2)}%", end="", flush=True)
            
            # Get current end effector pose
            current_pose = self.kinematics.pose_from_tf(self.kinematics.tf_A07(self.joint_states.position))

            J = self.kinematics.compute_jacobian(self.joint_states.position)
            J = self.kinematics.numerical_jacobian(self.joint_states.position)
            pinvJ  = np.linalg.pinv(J)
      
            # position and orientation error
            X[:3] = P[i] - current_pose[:3]
            X[3:] = O[i] - current_pose[3:]

            # compute the angular velocities
            if dt < 1e-6:
                Xdot = np.zeros(6)
            else:
                Xdot =  np.array([x/dt for x in X])

            alpha = 0.1
            k1 = np.array([1, 1, 1, 1, 1, 1, 1]).T

            # print("\n\nDimensions")
            # print(f"pinvJ: {pinvJ.shape}")
            # print(f"J: {J.shape}")
            # print(f"x:{X.shape}")
            # print(f"Xdot: {Xdot.shape}")
            # print(f"k1 : {k1.shape} ")
            
            self.joint_angvel  = pinvJ (X) #@ (kd @ Xdot + kp @ X) + (np.eye(7) - pinvJ @ J) @ k1 #kp @ X ) 

            time_prev =  self.time_now
            rostime_now = rospy.get_rostime()
            self.time_now = rostime_now.to_sec()
            dt = self.time_now - time_prev
            
            self.joint_angpos = np.add(self.joint_angpos, self.joint_angvel * dt)
            self.joint_angpos = np.asarray(self.joint_angpos).flatten()

            for j in range(7):
                self.joint_pos_pub[j].publish(self.joint_angpos[j])

            self.pub_rate.sleep()

        # Print progress completion if not in debug mode
        if not C_DEBUG_MODE:
            print("\r PROGRESS: Complete!", end="", flush=True)

        return True
    
    def full_control(self, n):
        
        self.joint_angpos = self.joint_states.position
        target_pose = np.zeros(6)
        target_velocity = np.zeros(6)
        kp = 0.2 * np.eye(6)
        time = 0
        
        X = np.zeros(6)
        Xdot = np.zeros(6)

        for i in range(0, n):
            if rospy.is_shutdown():
                return False

            if not C_DEBUG_MODE:
               print(f"\r PROGRESS: {round(i / n * 100.0, 2)}%", end="", flush=True)
            
            # Get current end effector pose
            current_pose = self.kinematics.pose_from_tf(self.kinematics.tf_A07(self.joint_states.position))
            
            J = self.kinematics.compute_jacobian(self.joint_states.position)
            #J = self.kinematics.numerical_jacobian(self.joint_states.position)
            pinvJ  = np.linalg.pinv(J)
            
            # get the desired position and velocity
            for j in range(0, 3):
                target_pose[j]        = self.a0_pos[j] + self.a1_pos[j] * time + self.a2_pos[j] * time**2  + self.a3_pos[j] * time**3
                target_pose[j+3]      = self.a0_ori[j] + self.a1_ori[j] * time + self.a2_ori[j] * time**2  + self.a3_ori[j] * time**3
                target_velocity[j]    =                  self.a1_pos[j] * time + self.a2_pos[j] * 2 * time + self.a3_pos[j] * 3 * time**2
                target_velocity[j+3]  =                  self.a1_ori[j] * time + self.a2_ori[j] * 2 * time + self.a3_ori[j] * 3 * time**2
            
            X = target_pose - current_pose
            
            # compute the angular velocities
            self.joint_angvel  = pinvJ @ ( X ) + (np.eye(7) - pinvJ @ J) @ np.array([1, 1, 1, 1, 1, 1, 1])

            time_prev =  self.time_now
            rostime_now = rospy.get_rostime()
            self.time_now = rostime_now.to_sec()
            dt = self.time_now - time_prev
            
            # update the time counter for the target position computation
            time = time + dt 

            self.joint_angpos = np.add(self.joint_angpos, self.joint_angvel * dt)
            self.joint_angpos = np.asarray(self.joint_angpos).flatten()

            for j in range(7):
                self.joint_pos_pub[j].publish(self.joint_angpos[j])

            self.pub_rate.sleep()

        # Print progress completion if not in debug mode
        if not C_DEBUG_MODE:
            print("\r PROGRESS: Complete!", end="", flush=True)

        return True
    
    def publish(self):
        """
        Implements the task 2:
        Full position and orientation control.
        """

        # Set initial configuration
        self.joint_angpos = [0, 0.75, 0, 1.5, 0, 0.75, 0]
        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()

        self.joint_pos_pub[3].publish(self.joint_angpos[3])
        tmp_rate.sleep()
        self.joint_pos_pub[1].publish(self.joint_angpos[1])
        self.joint_pos_pub[5].publish(self.joint_angpos[5])
        tmp_rate.sleep()

        # get the current pose
        cur_pose = np.zeros(6)
        cur_pose[:3] = self.kinematics.tf_A07(self.joint_states.position)[0:3,3]
        cur_pose

        # add the first waypoint as the current position of the robot
        P_task = []
        P_task.append(cur_pose)
        
        # read the user-specified waypoints file
        P_user = read_task_file(C_USER_TASK_FILE)
        P_task.extend(P_user)

        rostime_now = rospy.get_rostime()
        self.time_now = rostime_now.to_sec()

        if C_DEBUG_MODE:
            print(f"\nUser-defined waypoints:\n{P_user}")

        while not rospy.is_shutdown():
            for i in range(0, len(P_user)-1):
                
                P0, P1 = P_user[i][:3], P_user[i+1][:3]
                O0, O1 = P_user[i][3:], P_user[i+1][3:] 

                print(f"\n TASK: {P0}, {O0} ==> {P1}, {O1} \n", end="")
            
                n = 100
                self.a0_pos, self.a1_pos, self.a2_pos, self.a3_pos = cubic_interpolation(P0, P1, n, 1/self.rate)
                self.a0_ori, self.a1_ori, self.a2_ori, self.a3_ori = cubic_interpolation(O0, O1, n, 1/self.rate)  
                self.full_control(n)

                # P = linear_interpolation(P0, P1, self.rate)
                # O = linear_interpolation(O0, O1, self.rate)
                # self.full_control_with_linear_interpolation(P, O)

                self.print_current_ee_orientation("- After the task")
                self.print_current_ee_position("- After the task")

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
