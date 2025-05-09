#!/usr/bin/env python3
"""
Engineer    : Ronaldo Tsela
Date        : 18/4/2025
Description : This is a ROS node that implements a 3 DOF path following controller
            for the xArm7 cobot manipulator. This is part of the final project on
            Control and Robotics MSc program at NTUA for the academic year 2024/2025.
Note        : This script is an extended and refined version of the provided one.
"""

import sys
import rospy

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

import numpy as np

from kinematics import xArm7_kinematics

# Additional utility functions created
from utils import interpolate_points
from utils import read_task_file

C_USER_TASK_FILE = "task_file.txt"
C_DEBUG_MODE     = False             

class xArm7_controller():
    """
    Class to compute and publish joints positions
    """
    
    def __init__(self, rate):

        #-------- Robot Manipulator Setup ---------------
        # Initialize the xArm7 kinematics handler
        self.kinematics = xArm7_kinematics()

        # Initial joints' angular positions
        self.joint_angpos = [0.0,  0.0,  0.0, 0.0, 0.0, 0.0,  0.0]
        
        # Initial joints' states
        self.joint_states = JointState()

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

        # Start the main ROS loop
        self.publish()

    def joint_states_callback(self, msg):
        self.joint_states = msg

    # Main function that implements the control algorithm as reqested 
    # in part A of the project requirements
    def position_control_algorithm(self, P):
        """
        Implements a simple linear path following algorithm
        @param P: The set of waypoints the robot should pass
        @return : False if ROS server is off, True when the path is complete
        """
        # compute the joint positions for the target path through inverse kinematics
        Q = [self.kinematics.compute_angles(p) for p in P]

        # iterate over the joint positions
        for idx, joint_set in enumerate(Q):
            
            if not C_DEBUG_MODE:
                print(f"\r PROGRESS: {round(idx / len(Q) * 100.0, 2)}%", end="", flush=True)
            
            self.joint_angpos  = joint_set

            if rospy.is_shutdown():
                return False

            if C_DEBUG_MODE:
                self.print_target_ee_position(P[idx])
                self.print_current_ee_position(extra_msg="-Before Command")
                self.print_target_joint_position(self.joint_angpos)
                self.print_current_joint_position(extra_msg="-Before Command")
                self.print_target_fk_solution(self.joint_angpos)

            # publish the new joint's angular positions
            self.joint_pos_pub[0].publish(self.joint_angpos[0])
            self.joint_pos_pub[1].publish(self.joint_angpos[1])
            self.joint_pos_pub[3].publish(self.joint_angpos[3])
        
            # let it execute                    
            self.pub_rate.sleep()
            
            if C_DEBUG_MODE:
                self.print_current_ee_position(extra_msg="-After Command")

        if not C_DEBUG_MODE:
            print("\r PROGRESS: Complete!", end="", flush=True)
        
        return True

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
     
    def publish(self):
        """
        Implements the task 1:
        Position control on joints q1, q2 and q4 for a linear path 
        Requested the line with x = 0.6043 and z = 0.1508
        and points A and B set to 40 cm appart and symmetric along x-axis.
        However this script is capable of reading a user-defined high-level
        task description script.
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

        if C_DEBUG_MODE:
            self.print_current_joint_position(extra_msg="-After Initialization")
            self.print_current_ee_position(extra_msg="-After Initialization")
            
        # append the callculated positions to the current position
        cur_pos = []
        for p in self.kinematics.tf_A07(self.joint_states.position):
            cur_pos.append(p)
        
        # add the dummy orientation
        cur_pos.append(0.0)
        cur_pos.append(0.0)
        cur_pos.append(0.0)
        
        P_task = []
        P_task.append(cur_pos)
        
        # read the user-specified waypoints file
        P_user = read_task_file(C_USER_TASK_FILE)

        # append the user-specified waypoints in the current position
        P_task.extend(P_user)

        if C_DEBUG_MODE:
            print(f"\nUser-defined waypoints:\n{P_user}")

        while not rospy.is_shutdown():

            for i in range(0, len(P_user)-1):        
                # get a pair of points
                P0 = P_user[i][0:3]
                P1 = P_user[i+1][0:3]  

                # interpolate linearly
                P = interpolate_points(P0, P1, self.rate)
                
                print(f"\n TASK: {P0} ==> {P1} \n", end="")

                # command the robot to track the path
                self.position_control_algorithm(P)

    def turn_off(self):
        rospy.loginfo("Shutting down ROS")
        sys.exit(0)

def controller_py():
    rospy.init_node('controller_node', anonymous=True)
    
    rate       = rospy.get_param("/rate")
    controller = xArm7_controller(rate)
    
    rospy.on_shutdown(controller.turn_off)
    rospy.spin()

if __name__ == '__main__':
    try:
        controller_py()
    except rospy.ROSInterruptException:
        pass
