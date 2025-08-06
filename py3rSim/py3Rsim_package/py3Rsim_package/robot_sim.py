#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Homogeneous Transform Utilities -------------------
def homogeneous_transform(DH_table_row):
    # Extract the DH parameters.
    a     = DH_table_row[0]
    alpha = DH_table_row[1]
    d     = DH_table_row[2]
    theta = DH_table_row[3]

    # Create the individual transformation matrices.
    Rot_z = Rot("z", theta)
    Tra_z = Tra("z", d)
    Tra_x = Tra("x", a)
    Rot_x = Rot("x", alpha)
    
    # Combine them: A = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
    A = Rot_z @ Tra_z @ Tra_x @ Rot_x
    return A

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

# ----------------- 5-th Order Polynomial Interpolation Utilities -----
def poly5_interpolation(p0, v0, a0, pf, vf, af, tf):
    """
    Compute the coefficients of a 5th-order polynomial that satisfies:
    p(0)       = p0   ,   p(tf)       = pf
    dot p(0)   = v0   ,   dot p(tf)   = vf
    ddot p(0)  = a0   ,   ddot p (tf) = af

    :param p0: initial position
    :param v0: initial velocity
    :param a0: initial acceleration
    :param pf: final position
    :param vf: final velocity
    :param af: final acceleration
    :param tf: total time of the segment

    :return: [b0, b1, b2, b3, b4, b5] so that p(t) = b0 + b1 t + b2 t^2 + b3 t^3 + b4 t^4 + b5 t^5
    """
    # Known direct placements:
    b0 = p0
    b1 = v0
    b2 = a0 / 2.0

    # three unknowns remain: b3, b4, b5
    # they are computed from the boundary conditions at t = tf for p, dot p, ddot p
    # p(tf)      = b0 + b1 * tf + b2 * tf^2 + b3 * tf^3 + b4 * tf^4 + b5 * tf^5       = pf
    # dot p(tf)  = b1 + 2 * b2 * tf + 3 * b3 * tf^2 + 4 * b4 * tf^3 + 5 * b5 * tf^4   = vf
    # ddot p(tf) = 2 * b2 + 6 * b3 * tf + 12 * b4 * tf^2 + 20 * b5 * tf^3             = af

    # Construct system of equations for [b3, b4, b5]
    T2 = tf * tf
    T3 = T2 * tf
    T4 = T3 * tf
    T5 = T4 * tf

    M = np.array([
        [T3    ,       T4,       T5],
        [3 * T2,   4 * T3,   5 * T4],
        [6 * tf,  12 * T2,  20 * T3]
    ], dtype=float)

    rhs = np.array([
        pf - (b0 + b1 * tf + b2 * T2),       
        vf - (b1 + 2 * b2 * tf),             
        af - (2 * b2)                      
    ], dtype=float)

    # Solve for b3, b4, b5
    b3, b4, b5 = np.linalg.solve(M, rhs)

    return [b0, b1, b2, b3, b4, b5]

def evaluate_poly5(coeffs, t):
    """
    Evaluate the 5th-order polynomial at time t.
    :param coeffs : [b0, b1, b2, b3, b4, b5]
    :param : t time to evaluate at
    returns p(t) = b0 + b1 t + b2 t^2 + b3 t^3 + b4 t^4 + b5 t^5
    """
    b0, b1, b2, b3, b4, b5 = coeffs
    return (b0 + b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5)

def evaluate_poly5_vel(coeffs, t):
    """
    Evaluate the velocity (1st derivative) of the 5th-order polynomial.
    dot p(t) = b1 + 2 b2 t + 3 b3 t^2 + 4 b4 t^3 + 5 b5 t^4
    """
    b0, b1, b2, b3, b4, b5 = coeffs
    return (b1 + 2 * b2 * t + 3 * b3 * t**2 + 4 * b4 * t**3 + 5 * b5 * t**4)

def evaluate_poly5_acc(coeffs, t):
    """
    Evaluate the acceleration (2nd derivative) of the 5th-order polynomial.
    ddot p(t) = 2 b2 + 6 b3 t + 12 b4 t^2 + 20 b5 t^3
    """
    b0, b1, b2, b3, b4, b5 = coeffs
    return (2 * b2 +  6 * b3 * t + 12 * b4 * t**2 + 20 * b5 * t**3)

def generate_trajectory_5th_roder( waypoints, velocities, accelerations, position_limits, max_speed,  max_acceleration, tf_vector, T):
    """
    Given a list of waypoints, a list of velocities and a list of accelerations 
    to pass over each waypoint and the total time tf to perform on each segment 
    with time interval of T is computes the successive positions and composes the trajectory.
    :param waypoints : list of waypoints (each a 3-element list (x,y,z))
    :param velocities: list of velocities (each a 3-element list (vx,vy,vz))
    :param accelerations: list of accelerations (each a 3-element list (ax, ay, azz))
    :position_limits : list of a pair of limits for each coordinate ([min_x, max_x], [min_y, max_y], [min_z, max_z])
    :max_speed       : the maximum velocity permited in means of magnitude (norm(V)) for each segment
    :max_acceleration: the maximum acceleration permited in means of magnitude (norm(a)) for each segment
    :param tf        : total time for the motion per segment
    :return          : list of waypoints, velocities and accelerations at each point if valid.
    """
    is_valid = True
    R_lim, z_lim = position_limits

    trajectory_out   = []
    velocity_out     = []
    acceleration_out = []
 
    # For each segment between waypoint k and k+1
    for k in range(len(waypoints) - 1):

        tf = tf_vector[k]
        
        x0, y0, z0  = waypoints[k]
        x1, y1, z1  = waypoints[k + 1]

        vx0, vy0, vz0 = velocities[k]
        vx1, vy1, vz1 = velocities[k + 1]

        ax0, ay0, az0 = accelerations[k]
        ax1, ay1, az1 = accelerations[k + 1]

        # solve for polynomial coefficients in each dimension
        cx = poly5_interpolation(x0, vx0, ax0, x1, vx1, ax1, tf)
        cy = poly5_interpolation(y0, vy0, ay0, y1, vy1, ay1, tf)
        cz = poly5_interpolation(z0, vz0, az0, z1, vz1, az1, tf)

        # Local buffers
        seg_positions = []
        seg_vels      = []
        seg_accs      = []

        N = int(round(tf / T))  # number of points per segment

        for n in range(N):
            t = n * T # get this time instant

            # position
            x = evaluate_poly5(cx, t)
            y = evaluate_poly5(cy, t)
            z = evaluate_poly5(cz, t)
            
            r = np.sqrt(x**2 + y**2 + (z - z_lim)**2)
            if r > R_lim or z < 0:
                print(f"[WARNING] position out of range: ({x, y, z}) \n {r} /= {R_lim}")
                is_valid = False

            seg_positions.append([x, y, z])

            # velocity
            vx = evaluate_poly5_vel(cx, t)
            vy = evaluate_poly5_vel(cy, t)
            vz = evaluate_poly5_vel(cz, t)

            speed = np.linalg.norm([vx, vy, vz])

            if speed > max_speed[k]:
                print(f"[WARNING] speed exceeded: {speed} at segment {k}, t={t}")
                is_valid = False

            seg_vels.append([vx, vy, vz])

            # acceleration
            ax = evaluate_poly5_acc(cx, t)
            ay = evaluate_poly5_acc(cy, t)
            az = evaluate_poly5_acc(cz, t)

            acc_norm = np.linalg.norm([ax, ay, az])

            if acc_norm > max_acceleration[k]:
                print(f"[WARNING] accel exceeded: {acc_norm} at segment {k}, t={t}")
                is_valid = False

            seg_accs.append([ax, ay, az])

        # append to global arrays for output
        trajectory_out.extend(seg_positions)
        velocity_out.extend(seg_vels)
        acceleration_out.extend(seg_accs)

        if not is_valid:
            break

    return trajectory_out, velocity_out, acceleration_out, is_valid

# ----------------- 3R Robot Manipulator Sim Class ---------------------
class robot3RSim():
    def __init__(self, P_home=None, ax3d=None):
        """
        :param P_home: Home position for inverse kinematics (optional).
        :param ax3d  : A Matplotlib Axes3D object into which the robot will be drawn
                       (if None, a new figure and 3D axis are created)
        """
        # If the user did not defined a home point
        # use as default the initial configuration with
        # all joints set to zero in the theretical analysis
        if P_home is not None:
            self.P_home = P_home
        else:
            self.P_home = np.array([0.0, 0.0, 0.0])
        
        # set limits for joint movement (in radians)
        self.q1_range = (-np.pi, np.pi)
        self.q2_range = (-np.pi, np.pi)
        self.q3_range = (-np.pi, np.pi)

        # rad / s
        self.dq1_max = 3
        self.dq2_max = 3
        self.dq3_max = 3
        
        # set the base at origin
        self.base = np.array([0.0, 0.0, 0.0])

        # The manipulator geometrical properties (in meters)
        # these variables are private!
        self.__l1 = 0.42   # distance from base to joint 1
        self.__l2 = 0.45   # length of link 1 (joint 1 to joint 3)
        self.__l3 = 0.835  # length of link 2 (joint 3 to end effector)

        # Initialize the robot to the home position/configuration
        self.Q = self.inverse_kinematics(self.P_home)
        
        # intialize the robot with joints at rest
        self.dQ = np.array([0.0, 0.0, 0.0]) # rad/sec

        # set workspace limits
        self.R_lim = self.__l2 + self.__l3
        self.z_lim = self.__l1

        # If no axis was provided, create a new figure and axis
        if ax3d is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = ax3d
            self.fig = ax3d.figure  # The figure associated with the provided 3D axis
        
        # Create line objects for the links 
        self.link_bj1,  = self.ax.plot([], [], [], 'o-', lw=6, color='b')
        self.link_j1j2, = self.ax.plot([], [], [], 'o-', lw=5, color='g')
        self.link_j2j3, = self.ax.plot([], [], [], 'o-', lw=4, color='r')
        self.link_j3ee, = self.ax.plot([], [], [], 'o-', lw=3, color='m')

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 1.5)

    def DH_table(self, Q=None):
        """
        Construct the DH matrix based on the theoretical analysis.
        :param Q : the configuration vector [q1, q2, q3]
        returns the DH table matrix
        """
        if Q is not None:
            q1, q2, q3 = Q
        else:
            q1, q2, q3 = self.Q
         
        DH = np.array([
           #      a (m),    α (rad),        d (m),           θ (rad)   
            [   0.0,      np.pi/2,         self.__l1,          q1],            # base -> frame 1
            [self.__l2,        0.0,            0.0,     np.pi/2 + q2],           # frame 1 -> frame 2
            [   0.0,      np.pi/2,            0.0,     np.pi/2 + q3],           # frame 2 -> frame 3
            [   0.0,         0.0,           self.__l3,         0.0]                # frame 3 -> end effector
        ])

        return DH 

    def Jacobian_matrix(self, Q=None):
        """
        Compute the Jacobian matrix based on the theoretical analysis.
        :param Q: the configuration vector [q1, q2, q3]
        returns  the Jacobian matrix for linear velocities JL and angular velocities JA. 
        """
        if Q is not None:
            q1, q2, q3  = Q
        else:
            q1, q2, q3 = self.Q

        s1, c1      = np.sin(q1), np.cos(q1)
        s2, c2      = np.sin(q2), np.cos(q2)
        s23, c23    = np.sin(q2+q3), np.cos(q2+q3)

        # The linear velocity Jacobian
        JL = np.array([
            [-self.__l3 * s1 * s23 + self.__l2 * s1 * s2, -self.__l3 * c1 * c23 - self.__l2 * c1 * c2, -self.__l3 * c1 * c23],
            [-self.__l3 * c1 * s23 - self.__l2 * c1 * s2, -self.__l3 * s1 * c23 - self.__l2 * s1 * c2, -self.__l3 * s1 * c23],
            [            0.0                            , -self.__l3 * s23 - self.__l2 * s2          , -self.__l3 * s23     ]
        ])

        # The angular velocity Jacobian.
        # Note: Here this is not used!
        JA = np.array([
            [ 0.0,  s1,  s1],
            [ 0.0, -c1, -c1],
            [ 1.0, 0.0, 0.0]
        ])

        return JL, JA

    def forward_kinematics(self, Q=None):
        """
        Given a configuration [q1, q2, q3] for the 3R manipulator,
        return a 5x3 array where each row is the (x, y, z) position of:
          row 0: base,
          row 1: joint 1,
          row 2: joint 2,
          row 3: joint 3,
          row 4: end effector.
        """
        # update the DH table with the given configuration.
        # If None, use the internal state
        if Q is None:
            DH = self.DH_table()
        else:
            DH = self.DH_table(Q)
        
        # Compute the homogeneous local transform matrices
        A01 = homogeneous_transform(DH[0])
        A12 = homogeneous_transform(DH[1])
        A23 = homogeneous_transform(DH[2])
        A3E = homogeneous_transform(DH[3])
        
        # Compute the transforms from base to each subsequent frame
        T0_1 = A01                  # base -> frame 1
        T0_2 = T0_1 @ A12           # base -> frame 2
        T0_3 = T0_2 @ A23           # base -> frame 3
        T0_E = T0_3 @ A3E           # base -> end effector

        # Extract the Cartesian positions of each frame
        pos = np.zeros((5, 3))
        pos[0, :] = np.array(self.base) # base
        pos[1, :] = T0_1[0:3, 3]        # joint 1
        pos[2, :] = T0_2[0:3, 3]        # joint 2
        pos[3, :] = T0_3[0:3, 3]        # joint 3
        pos[4, :] = T0_E[0:3, 3]        # end effector

        return pos

    def inverse_kinematics(self, P):
        """
        Given the desired end-effector position P = (px, py, pz) (in meters),
        solve for the robot's joint angles [q1, q2, q3] (in rad).
        returns the joint configuration vector (3x1).
        """
        px, py, pz = P

        # Uesed for readability
        l0 = self.__l1    # offser along z
        l1 = self.__l2    # link 2 length
        l2 = self.__l3    # link 3 length

        # Compute the first joint angle from the x and y position
        q1 = np.arctan2(py, px)

        if q1 == np.pi/2 or q1 == -np.pi/2:
            q1 = q1 + (q1/abs(q1)) * np.deg2rad(1) # add to the direction of change 0.1 deggrees to avoid the zero

        # Convert the 3D problem into a 2D problem by performing a coordinate transformation
        x = px / np.cos(q1)  # horizontal distance in the plane
        z = pz - l0          # vertical distance
        r = x**2 + z**2 

        # Compute the elbow angle
        cos_q3 =  (r - l1**2 - l2**2) / (2 * l1 * l2)
        sin_q3 = np.sqrt(1 - cos_q3**2)
        q3     = -np.atan2(sin_q3, cos_q3)

        # Compute the shoulder angle
        sin_q2 = (x * (l1 + l2 * cos_q3) - z * l2 * sin_q3) / (l1**2 + l2**2 + 2 * l1 * l2 * cos_q3)
        cos_q2 = np.sqrt(1 - sin_q2**2)
        q2     = -np.atan2(sin_q2, cos_q2)

        Q = np.array([q1, q2, q3])

        # Note: I am using "-" for q3 and q2 because of the sign convention of the frames. 
        #       If q2 and q3 are used with "+" the robot will perform the motion from the opposite
        #       direction (like a mirror) to the actual positions set to track.

        return Q

    def forward_differential_kinematics(self, Q=None, dQ=None):
        """
        Solve the forward differential kinematics using the defined
        Jacobian matrix as analysed in the theoretical part.
        :param Q  : the configuration (q1, q2, q3) of the manipulator
        :param dQ : the joint velocities
        returns the linear and angular velocity vectors of the end effector
        """
        # Construct the Jacobian matrix for the current configuration
        if Q is not None:
            JL, JA  = self.Jacobian_matrix(Q)
        else: # use state
            JL, JA  = self.Jacobian_matrix()

        if dQ is not None:     
            v = JL @ dQ # Compute the 3x1 linear velocities vector
            w = JA @ dQ # Compute the 3x1 angular velocities vector
        else: # use state
            v = JL @ self.dQ # Compute the 3x1 linear velocities vector
            w = JA @ self.dQ # Compute the 3x1 angular velocities vector

        return v, w
    
    def generate_signals(self, trajectory, dt=0.001):
        """
        Generates the output control variables / signals that will drive the robot.
        : param trajectory : the trajectory designed for the robot to follow.
        : param dt         : time resolution. If not defined is set to 1 ms  
        returns the displacement of each joint per timestep and the velocities of each 
                joint per timestep respectivelly.              
        """
        q1_out, q2_out, q3_out      = [], [], []    # store the joint displacement
        dq1_out, dq2_out, dq3_out   = [], [], []    # store the joint velocities
        
        for point in trajectory:

            # compute the joint displacement
            q1, q2, q3 = self.inverse_kinematics(point)
            
            # Compute the joint speed
            dq1 = (q1 - self.Q[0])/dt
            dq2 = (q2 - self.Q[1])/dt
            dq3 = (q3 - self.Q[2])/dt

            # update the displacement state
            if self.update_q1(q1):
                print(f"[ERROR] Inapropriate value of q1 : {q1}")
                exit(-1)

            if self.update_q2(q2):
                print(f"[ERROR] Inapropriate value of q2 : {q2}")
                exit(-1)

            if self.update_q3(q3):
                print(f"[ERROR] Inapropriate value of q3 : {q3}")
                exit(-1)

            # Ckeck if the registered configuration leads to singularities
            v_sing, _ = self.check_singularities() # here we evaluate only linear velocity singularities!
            if v_sing:
                print(f"[ERROR] Singularities detected")
                exit(-1)

            # update the velocity state
            if self.update_dq1(dq1):   
                print(f"[ERROR] Inapropriate value of dq1 : {dq1}")
                exit(-1)

            if self.update_dq2(dq2):
                print(f"[ERROR] Inapropriate value of dq2 : {dq2}")
                exit(-1)

            if self.update_dq3(dq3):
                print(f"[ERROR] Inapropriate value of dq3 : {dq3}")
                exit(-1)

            # Log data       
            q1_out.append(q1)
            q2_out.append(q2)
            q3_out.append(q3)
            dq1_out.append(dq1)
            dq2_out.append(dq2)
            dq3_out.append(dq3)

        # Before returning the values re-initialize the state to the initial one
        # so we can use it for the simulation later
        self.dQ = np.array([0.0, 0.0, 0.0])
        self.Q  = self.inverse_kinematics(self.P_home)

        return q1_out, q2_out, q3_out, dq1_out, dq2_out, dq3_out
       
    def run(self, q1, q2, q3, dq1, dq2, dq3):
        """
        For each configuration vector and velocity vector update the internal joint configuration,
        compute the forward kinematics and end effector velocities with differential kinematics and
        animate the robot motion.
        :param q1, q2, q3    : the vector of displacement signals per time instant
        :param dq1, dq2, dq3 : the vector of joint speed signals per time instant
        returns the end effector position and the end effector linear velocities per timestep.
        """
        s = [] # store the position data for all frames of reference
               # this is used for the animation logic

        px_out, py_out, pz_out      = [], [], []    # store the end effector position
        vx_out, vy_out, vz_out      = [], [], []    # store the end effector velocities

        for i in range(len(q1)):

            # Update the robot's internal joint state (position and velocity)            
            self.update_q1(q1[i])
            self.update_q2(q2[i])
            self.update_q3(q3[i])
            self.update_dq1(dq1[i])
            self.update_dq2(dq2[i])
            self.update_dq3(dq3[i])

            # Compute the end effector velocities. If the velocity of 
            # each joint is grater than the maximum then use the maximum.
            # we should see the difference if any at the graphs!
            # We do not care for the angular velocities, only linear!
            v, _ = self.forward_differential_kinematics()

            # Compute the positions of each frame for that configuration
            # this is used for the animation logic
            pos = self.forward_kinematics()

            # Log data
            vx_out.append(v[0])
            vy_out.append(v[1])
            vz_out.append(v[2])  
            px_out.append(pos[4][0])
            py_out.append(pos[4][1])
            pz_out.append(pos[4][2])
            
            s.append(pos)
        
        # Animate the motion of the robot using the precomputed configurations
        self.animate_robot_motion(s)

        return px_out, py_out, pz_out, vx_out, vy_out, vz_out
    
    def check_singularities(self, Q=None):
        if Q is not None:
            JL, JL = self.Jacobian_matrix(Q)
        else:
            JL, JA = self.Jacobian_matrix()
        
        # compute the determinant of the Jacobian
        detJL = np.linalg.det(JL)
        detJA = np.linalg.det(JA)
        
        if detJL == 0:
            linear_singularities = True 
        else:
            linear_singularities = False

        if detJA == 0:
            angular_singularities = True
        else:
            angular_singularities = False 
        
        return linear_singularities, angular_singularities

    # ----------------- Internal state update functions -----------------
    def update_q1(self, q1):
        if q1 < self.q1_range[0] or q1 > self.q1_range[1]:
            print("[WARNING] q1 is out of range")
            return 1
        else:
            self.Q[0] = q1
            return 0

    def update_q2(self, q2):
        if q2 < self.q2_range[0] or q2 > self.q2_range[1]:
            print("[WARNING] q2 is out of range")
            return 1
        else:
            self.Q[1] = q2
            return 0

    def update_q3(self, q3):
        if q3 < self.q3_range[0] or q3 > self.q3_range[1]:
            print("[WARNING] q3 is out of range")
            return 1
        else:
            self.Q[2] = q3
            return 0

    def update_dq1(self, dq1):
        if abs(dq1) > self.dq1_max:
            print("[WARNING] dq1 is out of range")
            return 1
        else:
            self.dQ[0] = dq1
            return 0

    def update_dq2(self, dq2):
        if abs(dq2) > self.dq2_max:
            print("[WARNING] dq2 is out of range")
            return 1
        else:
            self.dQ[1] = dq2
            return 0
    
    def update_dq3(self, dq3):
        if abs(dq3) > self.dq3_max:
            print("[WARNING] dq3 is out of range")
            return 1
        else:
            self.dQ[2] = dq3
            return 0

    # ----------------- Visualization Utilities -----------------
    def add_link(self, link, P1, P2):
        x1, y1, z1 = P1
        x2, y2, z2 = P2
        link.set_data_3d([x1, x2], [y1, y2], [z1, z2])
 
    def animate_robot_motion(self, all_positions):
        """
        Animate the robot motion by drawing the links between the frames.
        param all_positions: list of 5x3 arrays (for each configuration).
        """
        # Animate the robot motion along the positions computed
        for pos in all_positions:
            # Extract positions of each frame
            pB  = pos[0]  # Base
            pJ1 = pos[1]  # Joint 1
            pJ2 = pos[2]  # Joint 2
            pJ3 = pos[3]  # Joint 3
            pEE = pos[4]  # End-Effector

            # Set the links between frames
            self.add_link(self.link_bj1, pB, pJ1)
            self.add_link(self.link_j1j2, pJ1, pJ2)
            self.add_link(self.link_j2j3, pJ2, pJ3)
            self.add_link(self.link_j3ee, pJ3, pEE)

            # Mark the end-effector and via point
            self.ax.plot([pEE[0]], [pEE[1]], [pEE[2]], 'k.', markersize=2, alpha=0.8)
           
            plt.draw()
            plt.pause(0.05)