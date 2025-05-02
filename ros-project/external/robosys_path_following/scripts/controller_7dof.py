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

# from tf.transformations import quaternion_matrix
# matrix = quaternion_matrix([1, 0, 0, 0])

C_USER_TASK_FILE = "task_file.txt"
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
    ###

    def full_control_algorithm(self, P=None, tolerance=1e-4):
        
        for i in range(0, len(P)-1):
            x0, y0, z0 = P[i]
            x1, y1, z1 = P[i+1]

            # Compute the time interval
            rostime_now = rospy.get_rostime()
            time_now    = rostime_now.to_sec()  

            # time interval
            dt = time_now - self.time_prev

            # update times
            self.time_prev = time_now 

            V = np.array([(x1 - x0)/dt, (y1 - y0)/dt, (z1 -z0)/dt], dtype=Float64)
            
            print(f"INFO: V = {V}")

            if not C_DEBUG_MODE:
                print(f"\r PROGRESS: {round(i / len(P) * 100.0, 2)}%", end="", flush=True)

            # Compute each transformation matrix wrt the base frame from joints' angular positions
            # self.A07 = self.kinematics.tf_A07(self.joint_angpos)
            # A07_real = self.kinematics.tf_A07(self.joint_states.position)

            # ee_pos = A07_real[0:3,3]
            # ee_ori = self.kinematics.rotationMatrixToEulerAngles( A07_real[0:3,0:3] )

            # Compute jacobian matrix
            # J = self.kinematics.compute_jacobian(self.joint_angpos)

            # # pseudoinverse jacobian
            # pinvJ = np.linalg.pinv(J)

            # """
            # INSERT YOUR MAIN CODE HERE
            # """

            # integrate angular velocity to get angular position
            # self.joint_angpos += self.joint_angvel * dt

            if rospy.is_shutdown():
                return False
            
            if C_DEBUG_MODE:
                self.print_target_ee_position(P[i])
                self.print_current_ee_position(extra_msg="-Before Command")
                self.print_target_joint_position(self.joint_angpos)
                self.print_current_joint_position(extra_msg="-Before Command")
                self.print_target_fk_solution(self.joint_angpos)

            # Publish the new joint's angular positions
            for i in range(0, 7):
                self.joint_pos_pub[i].publish(self.joint_angpos[i])
            
            # let it execute
            self.pub_rate.sleep()

            # control loop
            while True:
                
                # feedback
                q1, q2, _, q4    = self.joint_states.position[0:4]
                qc1, qc2, _, qc4 = self.joint_angpos[0:4]

                # compute the error from the target
                q1_error = abs(qc1 - q1)
                q2_error = abs(qc2 - q2)
                q4_error = abs(qc4 - q4)
                
                error_max = max([q1_error, q2_error, q4_error])

                if C_DEBUG_MODE:
                    print(f"q1 Error : {q1_error} | q2 Error : {q2_error} | q3 Error : {q4_error}")
                
                # check the error threshold
                if error_max > tolerance:
                    self.pub_rate.sleep()
                else:
                    break
                
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

        # get the time instance
        rostime_now = rospy.get_rostime()
        time_now = rostime_now.nsecs

        # append the callculated positions to the current position
        P_task = []
        P_task.append(self.kinematics.tf_A07(self.joint_states.position))
        
        # read the user-specified waypoints file
        P_user = read_task_file(C_USER_TASK_FILE)

        # append the user-specified waypoints in the current position
        P_task.extend(P_user)

        if C_DEBUG_MODE:
            print(f"\nUser-defined waypoints:\n{P_user}")

        while not rospy.is_shutdown():

            for i in range(0, len(P_user)-1):

                P0 = P_user[i]
                P1 = P_user[i+1]
                
                P = interpolate_points(P0, P1, self.rate)
                
                print(f"\b TASK: {P0} ==> {P1} \n", end="")
                
                self.full_control_algorithm(tolerance=1e-5, P=P)
            
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
