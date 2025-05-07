import numpy as np

def cross_product(a, b):
    """
    Compute the cross product from 2 given space vectors a and b
    """
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]

# Foundamental transformation operations
def Rot(axis, angle):
    """
    Construct a 4x4 rotation matrix about a given axis.
    """
    R = np.eye(4)
    c = np.cos(angle)
    s = np.sin(angle)

    if axis == "x":
        R[1, 1] = c;  R[1, 2] = -s
        R[2, 1] = s;  R[2, 2] = c
    elif axis == "y":
        R[0, 0] = c;  R[0, 2] = s
        R[2, 0] = -s; R[2, 2] = c
    elif axis == "z":
        R[0, 0] = c;  R[0, 1] = -s
        R[1, 0] = s;  R[1, 1] = c
    else:
        print("[ERROR] Rot : Invalid axis. Returning identity.")
    return R

def Tra(axis, displacement):
    """
    Construct a 4x4 translation matrix along a given axis.
    """
    T = np.eye(4)
    if axis == "x":
        T[0, 3] = displacement
    elif axis == "y":
        T[1, 3] = displacement
    elif axis == "z":
        T[2, 3] = displacement
    else:
        print("[ERROR] Tra : Invalid axis. Returning identity.")
    return T

def interpolate_points(P0, P1, n):
    """
    This function is used to interpolate two points in space using linear interpolation 
    @param P0 : the initial position in (X, Y, Z)
    @param P1 : the final position in (X, Y, Z)
    @param n  : the number of interpolated values to produce
    @return   : returns the set of interpolated positions
    """
    P0 = np.array(P0, dtype=float)
    P1 = np.array(P1, dtype=float)

    linear_points    = []

    # main interpolation loop
    for t in np.linspace(0, 1, n + 2):
        linear_points.append(P0 + (P1 - P0) * t )

    return np.array(linear_points)

def read_task_file(filename):
    """
    This function reads and decodes the user-specified commands from the 
    given TXT file. The source code is a modification of the code used for 
    controlling the NVIDIA's myCobot Jetson Cobot in lab 4.
    @param filename: The file that contains the robot task.
    @return a list of coordinate points for the robot to follow.
    """
    try:
        with open(filename, "r") as f:
            ft_list = f.readlines() # read all lines in the dile
    except FileNotFoundError: 
        print("ERROR: The file was not found")
    except IOError:
        print("ERROR: An I/O error occured while trying to read the file")
    
    P = []
    for line in ft_list:
        
        if line.find("set_pose:") != -1:
            try:
                coords_str = line.split(": ")[1]
                x, y, z, rx, ry, rz = map(float, coords_str.split(","))
                P.append((x, y, z, rx, ry, rz))
            except (ValueError, IndexError) as e: 
                print("WARNING: Invalid line : ", e)
                continue 
        else:
            print("ERROR: This version implemets only one commad!")
    
    return P