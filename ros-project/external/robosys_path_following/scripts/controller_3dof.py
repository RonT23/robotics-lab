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
        self.home_position = [0.0, 0.0,  0.0, 0.0, 0.0, 0.75, 0.0]

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
        self.pub_rate = rospy.Rate(rate)

        # Start the main ROS loop
        self.publish()

    def joint_states_callback(self, msg):
        self.joint_states = msg

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
        print(f"\nComputed IK End Effector Position")
        ee_position = self.kinematics.tf_A07(position)[0:3, 3]
        label = ["x", "y", "z"]

        for i, p in enumerate(ee_position):
            value = np.asscalar(p) if hasattr(np, 'asscalar') else p.item()
            print(f"Pe{label[i]} : {value:.4f}")

    def path_following_algorithm(self, P=None, tolerance=1e-3):
        """
        Implements a simple linear path following algorithm
        @param P: The set of waypoints the robot should pass
        @return : False if ROS server is off, True when the path is complete
        """
        # here I will add a linear interpolation algorithm function call!
        # for now assume constant path for testing
        # overwrite the given values of P!
        P = [ (0.5043, y, 0.15) for y in np.arange(-0.2, 0.2, 0.01)]
        
        # P = [
        #      (0.6043, -0.2, 0.15), 
        #      (0.6043, 0.3, 0.18), 
        #      (0.5043, -0.2, 0.20), 
        #      (0.6043, 0.1, 0.25)
        #     ]

        # compute the joint positions for the target path
        Q = [self.kinematics.compute_angles(p) for p in P]

        # iterate over the joint positions
        for idx, joint_set in enumerate(Q):
            
            self.joint_angpos  = joint_set

            if rospy.is_shutdown():
                return False

            # print the target end effector position ()
            self.print_target_joint_position(self.joint_angpos)
            self.print_target_ee_position(P[idx])

            # print the current joint positions (before the command)
            self.print_current_joint_position(extra_msg="-Before Command")
            self.print_current_ee_position(extra_msg="-Before Command")
            
            # publish the new joint's angular positions
            self.joint_pos_pub[0].publish(self.joint_angpos[0])
            self.joint_pos_pub[1].publish(self.joint_angpos[1])
            self.joint_pos_pub[3].publish(self.joint_angpos[3])
        
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

                print(f"\n q1 Error : {q1_error} rad")
                print(f"\n q2 Error : {q2_error} rad")
                print(f"\n q4 Error : {q4_error} rad")
                
                if error_max > tolerance:
                    self.pub_rate.sleep()
                else:
                    break

                break
            
            # print the current joint positions (after the command)
            self.print_current_joint_position(extra_msg="-After Command")
            self.print_current_ee_position(extra_msg="-After Command")

            # compute the forward kinematics using the target angle configuration
            self.print_target_fk_solution(self.joint_angpos)

        return True

    def publish(self):
        """
        Implements the task 1:
        Position control on joints q1, q2 and q4 for a linear path on x = 0.6043 and z = 0.1508
        The points A and B are set to 40 cm appart and symmetric along x-axis.
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

        rospy.loginfo("Device is set to initial configuration")
        rospy.loginfo("The system is ready to execute the path-following algorithm")

        self.print_current_joint_position(extra_msg="-After Initialization")
        self.print_current_ee_position(extra_msg="-After Initialization")
            
        # flag that indicates termination of the algorithm
        target_reached = False 
        while not rospy.is_shutdown():
            
            # execution of the algorithm
            if not target_reached:
                target_reached = self.path_following_algorithm(tolerance=0.015)
            else:
                break
 
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

