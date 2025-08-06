#!/usr/bin/env python3

import numpy as np

def task():
    # Define the path through waypoints
    waypoints = np.array([(0.45, 0.8,  1.2), 
                          (0.45, 0.1,  0.5),
                          (0.45, 0.1,  0.01)
                        ])
   
    # For each waypoint provide the velocity with which the 
    # robot will have at that particular point
    velocities = np.array([(0, 0, 0), 
                           (0, 0, -1),
                           (0, 0, 0)
                          ])
    
    # For each waypoint provide the acceleration of the end effector at 
    # that particular point
    accelerations = np.array([(0, 0, 0),  
                              (0, 0, 0),
                              (0, 0, 0)
                            ])
    
    # For each segment (point to point) provide a time for the end effector to move
    tf    = np.array([1.5, 1.5])  
    
    # For each segment provide the maximum permited speed (m/s) for the end effector
    max_v = np.array([5.0, 5.0])  

    # For each segment provide the maximum permited acceleration (m/s^2) for the end effector
    max_a = np.array([5.0, 5.0])  
      
    return waypoints, velocities, accelerations, tf, max_v, max_a