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
    """Class to compute and publish joints positions"""
    def __init__(self,rate):

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

        # Set the publishing rate
        self.period   = 1.0 / rate
        self.pub_rate = rospy.Rate(rate)

        # Start the main ROS loop
        self.publish()

    def joint_states_callback(self, msg):
        self.joint_states = msg

    def home(self):
        """Performs the homming operation of the robotic manipulator"""
        # Set the roboti to the initial configuration
        tmp_rate = rospy.Rate(1)

        for i in [3, 1, 5]:
            self.joint_pos_pub[i].publish(self.joint_angpos[i])
            tmp_rate.sleep()

    def publish(self):
        """
        Implements the task 1:
        Position control on joints q1, q2 and q4 for a linear path on x = 0.6043 and z = 0.1508
        The points A and B are set to 40 cm appart and symmetric along x-axis.
        """
        rospy.loginfo("Setting device to home")

        self.home()

        rospy.loginfo("The system is ready to execute the path-following algorithm")

        P = [ (0.6043, 0.2, 0.1508),
              (0.6043, -0.2, 0.1508)
        ]

        # First the robot must drive from current position to target position A
        # Next the robot must drive from position A to intermediate positions
        # as produced by a linear interpolation algorithm
        # From point-to-point we perform position control, 
        # impling that the current position compared with the target

        next_pos_index = 0

        rospy.loginfo(f"The first target is at : {P[next_pos_index]}")

        while not rospy.is_shutdown():
            
            # Compute each transformation matrix wrt the base frame from joints' angular positions
            self.A01 = self.kinematics.tf_A01(self.joint_angpos)
            self.A02 = self.kinematics.tf_A02(self.joint_angpos)
            self.A03 = self.kinematics.tf_A03(self.joint_angpos)
            self.A04 = self.kinematics.tf_A04(self.joint_angpos)
            self.A05 = self.kinematics.tf_A05(self.joint_angpos)
            self.A06 = self.kinematics.tf_A06(self.joint_angpos)
            self.A07 = self.kinematics.tf_A07(self.joint_angpos)
            
            # Compute the current position from the kinematic equation
            Pcurrent = self.A07[0:3,3]
            Pnext    = P[next_pos_index]
            
            # Compute the joint position of the target 
            self.joint_angpos = self.kinematics.compute_angles(Pnext)
            
            if np.allclose(Pcurrent, Pnext, atol=1e-4): # tolerance of 0.0004 m of difference
                next_pos_index += 1
                rospy.loginfo(f"Setting the next target position at : {P[next_pos_index]}")

            #for i in range(0, 7):
            #    print(self.joint_states.position[i])

            # Publish the new joint's angular positions
            for i in range(0, 7):
                self.joint_pos_pub[i].publish(self.joint_angpos[i])

            if np.allclose(Pcurrent, Pnext, atol=1e-4) and next_pos_index == len(P):
                # the taks is complete, break the loop
                rospy.loginfo("Task complete")
                break

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
