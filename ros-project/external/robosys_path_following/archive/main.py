#!/usr/bin/env python

import rospy
import time as t
import math
import numpy as np

import jacobian as J
from std_msgs.msg import Uint8
from geometry_msgs.msg import PointStamped

jp1_pub = rospy.Publisher('/mt_position_topic', position, queue_size=1)

def forward_kinematics_client(id):
    rospy.wait_for_service('forward_kinematics_service')
    try:
        fk_srv = rospy.ServiceProxy('forward_kinematics_service', fk_)
        response = fk_srv(id)
        return response.position
    except rospy.ServiceException, e:
        print "Failed to call service for Function-Approximation process: %s"%e

"""
def vel_control_callback(msg):
    sms.moveWithVelocity(msg.id, msg.value)

def handle_vel_encoder(req):
    vel_encoder_res = []
    for motor_id in range(num_of_motors):
        tmp = sms.getVelocity(motor_id_bias + motor_id)[1]
        tmp = -(tmp & 0x80000000) | (tmp & 0x7fffffff)
        vel_encoder_res.append(int(tmp))
    return vel_encoderResponse(vel_encoder_res)
"""

def redundant():
    id = Uint8()


    jp1_pub.publish(tmp)

def main_py():
    # Starts a new node
    rospy.init_node('main_node', anonymous=True)

    #port = rospy.get_param("~index_of_USB_port") # '/dev/ttyUSB_' (i.e. 0)

    #rospy.Subscriber("target", target, target_callback)

    #s_vel = rospy.Service('vel_encoder_service', vel_encoder, handle_vel_encoder)

    while not rospy.is_shutdown():
        redundant()
        #rospy.spin()

if __name__ == '__main__':
    try:
        main_py()
    except rospy.ROSInterruptException:
        pass
