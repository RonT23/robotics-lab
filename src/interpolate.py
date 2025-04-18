#!/bin/python3
"""
Designer 	: Ronaldo Tsela
Date     	: 10/4/2025
Description 	: Interpolate between two points in space using linear, quadratic or cubic interpolation
 	          algorithms
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def interpolate_points(P0, P1, n):
    """
    This function is used to interpolate two points in space using linear, quadratic and cubic interpolation
    methods. 
    @param P0 : the initial position in (X, Y, Z)
    @param P1 : the final position in (X, Y, Z)
    @param n  : the number of interpolated values to produce
    @return   : returns a dictionary containing the fields "linear", "quadratic" and "cubic". Each
    		field contains the values (X, Y, Z) of the interpolated aproximation. 
    """
    P0 = np.array(P0, dtype=float)
    P1 = np.array(P1, dtype=float)

    height = np.linalg.norm(P1 - P0) * 0.1
    control_quad = (P0 + P1) / 2 + np.array([0, 0, height])
    control_quad = np.array(control_quad, dtype=float)
   
    curve_height = np.linalg.norm(P1 - P0) * 0.1
    control_cubic1 = P0 + (P1 - P0) / 3 + np.array([0, 0, curve_height])
    control_cubic2 = P0 + 2 * (P1 - P0) / 3 + np.array([0, 0, -curve_height])
    control_cubic1 = np.array(control_cubic1, dtype=float)
    control_cubic2 = np.array(control_cubic2, dtype=float)
    
    linear_points    = []
    quadratic_points = []
    cubic_points     = []
    
    # main interpolation loop
    for t in np.linspace(0, 1, n + 2):
        linear_points.append(P0 + (P1 - P0) * t )
        quadratic_points.append( (1 - t)**2 * P0 + 2 * (1 - t) * t * control_quad + t**2 * P1 )
        cubic_points.append( (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * control_cubic1 + 3 * (1 - t) * t**2 * control_cubic2 + t**3 * P1 )

    return {
        "linear": np.array(linear_points),
        "quadratic": np.array(quadratic_points),
        "cubic": np.array(cubic_points)
    }

def write_points_to_file(filename, points):
    with open(filename, "w") as f:
         for point in points:
             x, y, z = point
             f.write(f"{x:.1f}, {y:.1f}, {z:.1f}\n")

if __name__ == "__main__":
	
	P0 = (-42.7, -232.3, 115.1)	# initial position
	P1 = (-147.1, -172.2, 117.7)	# final position
	n  = 4			# number of interpolated values
	
	# receive command line user input
	# if len(sys.argv) != 9:
	#     print(f"Usage : python3 {sys.argv[0]} filename x0 y0 z0 x1 y1 z1 n")
	#     sys.exit(1)
	
	# # filter input
	# x0, y0, z0, x1, y1, z1, n = map(float, sys.argv[2:])
	# filename = sys.argv[1]
	filename = "output"
     
	# P0 = (x0, y0, z0)
	# P1 = (x1, y1, z1)
	# n  = int(n)
	
	# interpolate
	result = interpolate_points(P0, P1, n)
	
	# prepare results
	linear_interp_res    = result["linear"]
	quadratic_interp_res = result["quadratic"]
	cubic_interp_res     = result["cubic"]
	
	# export results	
	write_points_to_file(f"{filename}_linear", linear_interp_res)
	write_points_to_file(f"{filename}_quadratic", quadratic_interp_res)
	write_points_to_file(f"{filename}_cubic", cubic_interp_res)
	
	# visualize results
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.plot(*result['linear'].T, label='Linear', color='blue')
	ax.plot(*result['quadratic'].T, label='Quadratic', color='green')
	ax.plot(*result['cubic'].T, label='Cubic', color='red')

	ax.scatter(*result['linear'].T, s=50, color='blue')
	ax.scatter(*result['quadratic'].T, s=50, color='green')
	ax.scatter(*result['cubic'].T, s=50, color='red')
	
	ax.scatter(*P0, color='black', s=50, label='Start (P0)')
	ax.scatter(*P1, color='black', s=50, label='End (P1)')

	ax.set_xlabel("X (mm)")
	ax.set_ylabel("Y (mm)")
	ax.set_zlabel("Z (mm)")
	ax.legend()
	ax.grid(True)
	plt.tight_layout()
	plt.show()
