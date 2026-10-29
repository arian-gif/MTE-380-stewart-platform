import numpy as np
# Kinematics from StewartPlatform_Slides.pdf. The "l" argument is unused (kept to match the slides).
# Forward functions take angles in degrees; inverse functions return angles in degrees.


def calculate_position_1(l, l_1, l_11, l_12, theta_11,theta_12):
    x1 = np.array([0,l_1+l_11*np.cos(theta_11*np.pi/180)-l_12*np.cos(
        theta_12*np.pi/180),l_11*np.sin(theta_11*np.pi/180)+l_12*np.sin(
        theta_12*np.pi/180)])
    return {"x1": x1}
def calculate_vectors_and_angles_1(l,l_1,l_11,l_12,x1,x2,x3):
    # Calculating theta_11
    # Calculate numerator
    AA_tmp = l_12**2-(l_1-x1[1])**2-l_11**2-x1[2]**2
    #Calculate denominator
    bb_tmp = (2 * (l_1 - x1[1]) * l_11)
    cc_tmp = (2 * x1[2] * l_11)
    denom_AA = np.sqrt(bb_tmp**2 + cc_tmp**2)
    #Calculate (BB + theta_11)
    cc = np.arccos((AA_tmp / denom_AA))
    #Calculate BB
    bb1 = np.arccos(bb_tmp / denom_AA)
    bb2 = np.arcsin(cc_tmp/ denom_AA)
    #Calculate theta_11
    theta_11 = cc - bb1
    #Calculate theta_12 (slide 20: tan(theta_12) = (z1 - l11 sin(theta_11)) / (l1 - y1 + l11 cos(theta_11)))
    #arctan2 instead of arcsin so the link can lean outward (theta_12 > 90 deg)
    theta_12 = np.arctan2(x1[2]-l_11*np.sin(theta_11), l_1-x1[1]+l_11*np.cos(theta_11))
    return {"theta_11": np.degrees(theta_11), "theta_12": np.degrees(theta_12)}


def calculate_position_2(l, l_2, l_21, l_22, theta_21, theta_22):
    ixy = l_2 + l_21*np.cos(theta_21*np.pi/180) - l_22*np.cos(theta_22*
        np.pi/180)
    iz = l_21*np.sin(theta_21*np.pi/180) + l_22*np.sin(theta_22*np.pi
        /180)
    x2 = np.array([ixy*np.cos(30*np.pi/180),-ixy*np.sin(30*np.pi/180),iz
        ])
    return {"x2": x2 }
def calculate_vectors_and_angles_2(l,l_2,l_21,l_22, x1,x2,x3):
    # Calculating theta_21
    #Calculate numerator in pieces
    y_tmp = np.sqrt(x2[0]**2+x2[1]**2)
    z_tmp = x2[2]
    AA_tmp =-(l_22**2-(l_2-y_tmp)**2-l_21**2-z_tmp**2)
    bb_tmp = (2 * (l_2 - y_tmp) * l_21)
    cc_tmp = (2 * z_tmp * l_21)
    #Calculate denominator
    denom_AA = np.sqrt(bb_tmp**2 + cc_tmp**2)
    #Compute BB-theta_21
    cc = np.arcsin(AA_tmp / denom_AA)
    #Compute BB
    bb1 = np.arcsin(bb_tmp / denom_AA)
    bb2 = np.arccos(cc_tmp/ denom_AA)
    #Compute theta_21 and theta_22
    theta_21 = cc + bb1
    #arctan2 instead of arcsin so the link can lean outward (theta_22 > 90 deg)
    theta_22 = np.arctan2(x2[2]-l_21*np.sin(theta_21), l_2-y_tmp+l_21*np.cos(theta_21))
    return {"theta_21": np.degrees(theta_21), "theta_22": np.degrees(theta_22)}


def calculate_position_3(l, l_3, l_31, l_32, theta_31, theta_32):
    ixy = l_3 + l_31*np.cos(theta_31*np.pi/180) - l_32*np.cos(theta_32*
        np.pi/180)
    iz = l_31*np.sin(theta_31*np.pi/180) + l_32*np.sin(theta_32*np.pi
        /180)
    x3 = np.array([-ixy*np.cos(30*np.pi/180),-ixy*np.sin(30*np.pi/180),
        iz])
    return {"x3": x3}
def calculate_vectors_and_angles_3(l,l_3,l_31,l_32,x1,x2,x3):
    # Calculating theta_31
    #Compute the numerator in pieces
    y_tmp = np.sqrt(x3[0]**2+x3[1]**2)
    z_tmp = x3[2]
    AA_tmp = l_32**2-(l_3-y_tmp)**2-l_31**2-z_tmp**2
    bb_tmp = (2 * (l_3 - y_tmp) * l_31)
    cc_tmp = (2 * z_tmp * l_31)
    #Compute denominator
    denom_AA = np.sqrt(bb_tmp**2 + cc_tmp**2)
    #Compute BB-theta_31
    cc = np.arccos(AA_tmp / denom_AA)
    #Compute BB
    bb1 = np.arccos(bb_tmp / denom_AA)
    bb2 = np.arcsin(cc_tmp/ denom_AA)
    #Compute theta_31 and theta_32
    #The slide's "cc - bb2" only works when the target is inside the motor radius (y_tmp < l_3).
    #"cc - bb1" is the same formula as actuator 1 and works for any target.
    theta_31 = cc - bb1
    #arctan2 instead of arcsin so the link can lean outward (theta_32 > 90 deg)
    theta_32 = np.arctan2(z_tmp-l_31*np.sin(theta_31), l_3-y_tmp+l_31*np.cos(theta_31))
    return {"theta_31": np.degrees(theta_31), "theta_32": np.degrees(theta_32)}
