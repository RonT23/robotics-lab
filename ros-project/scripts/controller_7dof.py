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
from utils import read_task_file
from utils import eulerAnglesToAngularVelocities

C_USER_TASK_FILE = "task_file_2.txt"
C_EVALUATION     = True 
C_EVALUATION_CYCLES = 10

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
    
    ## These functions are used for debugging
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

    ### This function implements the task 2
    def full_control(self, n):
        """
        """
        self.joint_angpos = self.joint_states.position
        target_pose = np.zeros(6)
        target_velocity = np.zeros(6)
        time = 0
        
        X = np.zeros(6)
        Xdot = np.zeros(6)

        kp = np.diag([5, 5, 5, 1.2, 1, 1])
        kr = np.array([0, 0, 0, 0, 0, 0, 0])
        knull = -np.eye(7) * 0.5 

        for i in range(0, n):
            if rospy.is_shutdown():
                return False

            print(f"\r PROGRESS: {round(i / n * 100.0, 2)}%", end="", flush=True)
            
            # Get current end effector pose
            current_pose = self.kinematics.pose_from_tf(self.kinematics.tf_A07(self.joint_states.position))
            
            J = self.kinematics.compute_jacobian(self.joint_angpos)
            pinvJ  = np.linalg.pinv(J)
            
            # get the desired position and velocity
            for j in range(0, 3):
                target_pose[j]        = self.a0_pos[j] + self.a1_pos[j] * time + self.a2_pos[j] * time**2  + self.a3_pos[j] * time**3
                target_pose[j+3]      = self.a0_ori[j] + self.a1_ori[j] * time + self.a2_ori[j] * time**2  + self.a3_ori[j] * time**3
                target_velocity[j]    =                  self.a1_pos[j] * time + self.a2_pos[j] * 2 * time + self.a3_pos[j] * 3 * time**2
                target_velocity[j+3]  =                  self.a1_ori[j] * time + self.a2_ori[j] * 2 * time + self.a3_ori[j] * 3 * time**2
            
            # for statistics
            self.x_inter.append(target_pose[0])
            self.y_inter.append(target_pose[1])
            self.z_inter.append(target_pose[2])
            
            self.thx_inter.append(target_pose[3])
            self.thy_inter.append(target_pose[4])
            self.thz_inter.append(target_pose[5])
            
            X = target_pose - current_pose
            Xdot[:3] = target_velocity[:3]
            #Xdot[3:] = target_velocity[3:]

            Xdot[3:] = eulerAnglesToAngularVelocities(target_pose[3:], target_velocity[3:])
            
            q = np.array(self.joint_states.position)
            q_desired = np.array(self.joint_angpos)
            kr = -knull @ (q - q_desired)
            
            # compute the angular velocities
            self.joint_angvel  = pinvJ @ ( Xdot + kp @ X ) + (np.eye(7) - pinvJ @ J) @ kr

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

            # record data for evaluation
            A07 = self.kinematics.tf_A07(self.joint_states.position)
            robot_pose = self.kinematics.pose_from_tf(A07)

            self.x_robot.append(robot_pose[0])
            self.y_robot.append(robot_pose[1])
            self.z_robot.append(robot_pose[2])

            self.thx_robot.append(robot_pose[3])
            self.thy_robot.append(robot_pose[4])
            self.thz_robot.append(robot_pose[5])
            
            self.q1_commanded.append(self.joint_angpos[0])
            self.q2_commanded.append(self.joint_angpos[1])
            self.q3_commanded.append(self.joint_angpos[2])
            self.q4_commanded.append(self.joint_angpos[3])
            self.q5_commanded.append(self.joint_angpos[4])
            self.q6_commanded.append(self.joint_angpos[5])
            self.q7_commanded.append(self.joint_angpos[6])
            
            self.q1_actual.append(self.joint_states.position[0])
            self.q2_actual.append(self.joint_states.position[1])
            self.q3_actual.append(self.joint_states.position[2])
            self.q4_actual.append(self.joint_states.position[3])
            self.q5_actual.append(self.joint_states.position[4])
            self.q6_actual.append(self.joint_states.position[5])
            self.q7_actual.append(self.joint_states.position[6])
        
            
        # Print progress completion if not in debug mode
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

        # used for debugging
        self.x_inter, self.y_inter, self.z_inter = [], [], []
        self.thx_inter, self.thy_inter, self.thz_inter = [], [], []

        self.thx_robot, self.thy_robot, self.thz_robot = [], [], []
        self.x_robot, self.y_robot, self.z_robot = [], [], []
        self.q1_actual, self.q2_actual, self.q3_actual = [], [], []
        self.q4_actual, self.q5_actual, self.q6_actual, self.q7_actual = [], [], [], []
        self.q1_commanded, self.q2_commanded, self.q3_commanded = [], [], []
        self.q4_commanded, self.q5_commanded, self.q6_commanded, self.q7_commanded = [], [], [], []

        iteration = 0

        while not rospy.is_shutdown():

            for i in range(0, len(P_user)-1):
                
                P0, P1 = P_user[i][:3], P_user[i+1][:3]
                O0, O1 = P_user[i][3:], P_user[i+1][3:] 

                print(f"\n TASK: {P0}, {O0} ==> {P1}, {O1} \n", end="")
      
                self.a0_pos, self.a1_pos, self.a2_pos, self.a3_pos = cubic_interpolation(P0, P1, self.rate, 1/self.rate)
                self.a0_ori, self.a1_ori, self.a2_ori, self.a3_ori = cubic_interpolation(O0, O1, self.rate, 1/self.rate)  
                
                self.full_control(self.rate)

                self.print_current_ee_orientation("- After the task")
                self.print_current_ee_position("- After the task")

            if C_EVALUATION: 
                iteration += 1
                if iteration == C_EVALUATION_CYCLES:
                    break 
        
        if C_EVALUATION:
            import matplotlib.pyplot as plt

            colors = [
                'tab:blue',   
                'tab:orange',
                'tab:green',  
                'tab:red',     
                'tab:purple',  
                'tab:brown',  
                'tab:pink'  
            ]

            # plot the actual VS desired position
            plt.figure(figsize=(12, 8))  

            plt.plot(self.x_inter, linestyle='--', linewidth=2, color=colors[0], label='X (Desired)') 
            plt.plot(self.y_inter, linestyle='--', linewidth=2, color=colors[1], label='Y (Desired)')
            plt.plot(self.z_inter, linestyle='--', linewidth=2, color=colors[2], label='Z (Desired)')

            plt.plot(self.x_robot, linestyle='-', linewidth=2, color=colors[0], label='X (Executed)') 
            plt.plot(self.y_robot, linestyle='-', linewidth=2, color=colors[1], label='Y (Executed)')
            plt.plot(self.z_robot, linestyle='-', linewidth=2, color=colors[2], label='Z (Executed)')

            plt.xlabel('Time Instance', fontsize=14) 
            plt.ylabel('End Effector Position (m)', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.tight_layout()  

            # plot the actual VS desired position
            plt.figure(figsize=(12, 8))  

            plt.plot(self.thx_inter, linestyle='--', linewidth=2, color=colors[0], label='Θx (Desired)') 
            plt.plot(self.thy_inter, linestyle='--', linewidth=2, color=colors[1], label='Θy (Desired)')
            plt.plot(self.thz_inter, linestyle='--', linewidth=2, color=colors[2], label='θz (Desired)')

            plt.plot(self.thx_robot, linestyle='-', linewidth=2, color=colors[0], label='θx (Executed)') 
            plt.plot(self.thy_robot, linestyle='-', linewidth=2, color=colors[1], label='θy (Executed)')
            plt.plot(self.thz_robot, linestyle='-', linewidth=2, color=colors[2], label='θz (Executed)')

            plt.xlabel('Time Instance', fontsize=14) 
            plt.ylabel('End Effector Orientation (rad)', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.tight_layout()  

            # plot the total position error 
            plt.figure(figsize=(12, 8))  
            plt.plot(np.array(self.x_inter) - np.array(self.x_robot), linestyle='--', linewidth=2, color=colors[0], label='X Error') 
            plt.plot(np.array(self.y_inter) - np.array(self.y_robot), linestyle='--', linewidth=2, color=colors[1], label='Y Error')
            plt.plot(np.array(self.z_inter) - np.array(self.z_robot), linestyle='--', linewidth=2, color=colors[2], label='Z Error')
            plt.xlabel('Time Instance', fontsize=14) 
            plt.ylabel('End Effector Position Error (m)', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.tight_layout()  

            # plot the total position error 
            plt.figure(figsize=(12, 8))  
            plt.plot(np.array(self.thx_inter) - np.array(self.thx_robot), linestyle='--', linewidth=2, color=colors[0], label='θx Error') 
            plt.plot(np.array(self.thy_inter) - np.array(self.thy_robot), linestyle='--', linewidth=2, color=colors[1], label='θy Error')
            plt.plot(np.array(self.thz_inter) - np.array(self.thz_robot), linestyle='--', linewidth=2, color=colors[2], label='θz Error')
            plt.xlabel('Time Instance', fontsize=14) 
            plt.ylabel('End Effector Orientation Error (rad)', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.tight_layout()  

            # Joint configuration actual VS commanded
            plt.figure(figsize=(12, 8))  
            plt.plot(self.q1_commanded, linestyle='--', linewidth=2, color=colors[0], label='q1 (Desired)') 
            plt.plot(self.q2_commanded, linestyle='--', linewidth=2, color=colors[1], label='q2 (Desired)')
            plt.plot(self.q3_commanded, linestyle='--', linewidth=2, color=colors[2], label='q3 (Desired)')
            plt.plot(self.q4_commanded, linestyle='--', linewidth=2, color=colors[3], label='q4 (Desired)') 
            plt.plot(self.q5_commanded, linestyle='--', linewidth=2, color=colors[4], label='q5 (Desired)')
            plt.plot(self.q6_commanded, linestyle='--', linewidth=2, color=colors[5], label='q6 (Desired)')
            plt.plot(self.q7_commanded, linestyle='--', linewidth=2, color=colors[6], label='q7 (Desired)') 
            
            plt.plot(self.q1_actual, linestyle='-', linewidth=2, color=colors[0], label='q1 (Actual)')
            plt.plot(self.q2_actual, linestyle='-', linewidth=2, color=colors[1], label='q2 (Actual)')
            plt.plot(self.q3_actual, linestyle='-', linewidth=2, color=colors[2], label='q3 (Actual)') 
            plt.plot(self.q4_actual, linestyle='-', linewidth=2, color=colors[3], label='q4 (Actual)')
            plt.plot(self.q5_actual, linestyle='-', linewidth=2, color=colors[4], label='q5 (Actual)')
            plt.plot(self.q6_actual, linestyle='-', linewidth=2, color=colors[5], label='q6 (Actual)') 
            plt.plot(self.q7_actual, linestyle='-', linewidth=2, color=colors[6], label='q7 (Actual)')
        
            plt.xlabel('Time Instance', fontsize=14) 
            plt.ylabel('Joint displacement (rad)', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.tight_layout()  

            plt.show()

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
