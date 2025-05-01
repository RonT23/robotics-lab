from kinematics import xArm7_kinematics
import numpy as np

if __name__ == "__main__":
    kine = xArm7_kinematics()

    Pe = [0.2, 0.0, 0.4]
    print(f"Pe: {Pe}\n")

    Q = kine.compute_angles(Pe)
    print(f"Q - IK: {Q}\n")

    print(f"\n{kine.tf_A07(Q)}\n")

    P = kine.tf_A07(Q)[0:3,3]
    print(f"P - FW: {np.transpose(P)}\n")
