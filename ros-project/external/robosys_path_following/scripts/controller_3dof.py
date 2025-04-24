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
from numpy.linalg import inv, det, norm, pinv
import time as t

from kinematics import xArm7_kinematics

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
        
        # Position set for home
        self.home_position = [0.0, 0.75, 0.0, 1.5, 0.0, 0.75, 0.0]

        # Initial joints' states
        self.joint_states = JointState()

        # Transformation matrices of joint frames from base
        self.A01 = self.kinematics.tf_A01(self.joint_angpos)
        self.A02 = self.kinematics.tf_A02(self.joint_angpos)
        self.A03 = self.kinematics.tf_A03(self.joint_angpos)
        self.A04 = self.kinematics.tf_A04(self.joint_angpos)
        self.A05 = self.kinematics.tf_A05(self.joint_angpos)
        self.A06 = self.kinematics.tf_A06(self.joint_angpos)
        self.A07 = self.kinematics.tf_A07(self.joint_angpos)

        #-------- ROS Environment Setup ------------------
        
        # Subscribe to the ROS topic where the xArm7 publishes the joint state
        self.joint_states_sub = rospy.Subscriber('/xarm/joint_states', JointState, self.joint_states_callback, queue_size=1)
        
        # Setup topics from where xArm7 listens for joint position angles 
        self.joint_pos_pub = []
        for i in range(0, 7):
            self.joint_pos_pub.append( rospy.Publisher(f'/xarm/joint{i+1}_position_controller/command', Float64, queue_size=1) )

        # publishing period
        self.pub_period   = 1.0 / rate

        # Set the publishing rate
        self.pub_rate = rospy.Rate(2)

        # Start the main ROS loop
        self.publish()

        #self.manual_control()

    def joint_states_callback(self, msg):
        self.joint_states = msg

    def home(self):
        """
        Performs the homming operation of the robotic manipulator
        """
        for i in [3, 1, 5]:
            self.joint_pos_pub[i].publish(Float64(self.home_position[i]))
            self.pub_rate.sleep()

    def manual_control(self):
        rospy.loginfo("Manual joint control started. Enter 7 joint angles in radians separated by spaces.")
        rospy.loginfo("Example: 0.0 0.5 -0.5 1.0 0.0 -0.2 0.3")
        
        while not rospy.is_shutdown():
            try:
                input_str = input("Enter joint angles (q1 q2 q3 q4 q5 q6 q7): ")
                angles = list(map(float, input_str.strip().split()))
                if len(angles) != 7:
                    print("Please enter exactly 7 joint angles.")
                    continue

                self.joint_angpos = angles
                rospy.loginfo(f"Publishing joint angles: {self.joint_angpos}")

                for i in range(7):
                    self.joint_pos_pub[i].publish(Float64(self.joint_angpos[i]))

                self.pub_rate.sleep()

            except ValueError:
                print("Invalid input. Please enter 7 float numbers separated by spaces.")
            except KeyboardInterrupt:
                print("\nManual control terminated.")
                break

    def publish(self):
        """
        Implements the task 1:
        Position control on joints q1, q2 and q4 for a linear path on x = 0.6043 and z = 0.1508
        The points A and B are set to 40 cm appart and symmetric along x-axis.
        """

        # rospy.loginfo("Device is set to home position")
        rospy.loginfo("The system is ready to execute the path-following algorithm")

        # get the current configuration
        while len(self.joint_states.position) == 0:
            pass
        
        for i in range(0, 7):
            self.joint_angpos[i] = self.joint_states.position[i]
        
        # find the current possition
        self.A07 = self.kinematics.tf_A07(self.joint_angpos)
        pos_init = self.A07[0:3, 3]

        # Set of positions to follow. The first position is the current position!
        P = [
            # x    y    z
            pos_init,
            (0.6043, -2, 0.1508),
            (0.6043, -1, 0.1508),
            (0.6043, 0.0e-6, 0.1508),
            (0.6043, 1, 0.1508),
            (0.6043, 2, 0.1508)
        ]

        print(f"curent configuration : {self.joint_states.position}")

        # Compute the joint positions for each of the coordinates provided
        Q = [self.kinematics.compute_angles(p) for p in P]

        for step_idx, joint_set in enumerate(Q):
            if rospy.is_shutdown():
               break
               
            rospy.loginfo(f"Publishing step {step_idx + 1}: {joint_set}")
            self.joint_angpos = joint_set
           
            # Publish the new joint's angular positions
            for i in range(0, 7):
                self.joint_pos_pub[i].publish(Float64(self.joint_angpos[i]))

            self.pub_rate.sleep()

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
