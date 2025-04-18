% Engineer      : Ronaldo Tsela
% Date          : 17/4/2025
% Description   : Compute the forward kinematics of the myCobot 280 Jetson
% Nano robot manipulator as part of the lab 4 in Robot Control Lab - NTUA -
% MSc Control and Robotics. 

clc; clear all; close all;

syms q1 q2 q3 q4 q5 q6

% The DH parameters of myCobot 280 
% for q4, q5, q6 constant to zero
q4 = 0; q5 = 0; q6 = 0;

% define the angles for the q1, q2, q3 joints
% comment for symbolic solution!
q1 = deg2rad(30);
q2 = deg2rad(-55);
q3 = deg2rad(50);

DH_params = [ ...
    q1,           131.22,   0.0,    pi/2;
    q2 + pi/2,    0.0,      110.4,  0.0;
    q3,           0.0,      96.0,   0.0;
    q4 - pi/2,    63.4,     0.0,   -pi/2;
    q5 - pi/2,    75.05,    0.0,   -pi/2;
    q6,           45.6,     0.0,    0.0;
];

% Compute the forward kinematics equation
myCobot280_Jetson_Nano_kinematic_eq = Forward_Kinematics(DH_params)

%%====== Functions
function kinematic_equation = Forward_Kinematics(DH_parameter_table) 
    % Transformations employed for the homogeneous transform with DH
    Rot_x = @(a) [ 1    0       0      0;
                   0 cos(a) -sin(a)    0;
                   0 sin(a)  cos(a)    0;
                   0    0       0      1 ];
      
    Rot_z = @(a) [ cos(a)  -sin(a)  0  0;
                   sin(a)   cos(a)  0  0;
                      0       0     1  0;
                      0       0     0  1 ];
    
    Tra_x = @(d) [ 1  0  0  d;
                   0  1  0  0;
                   0  0  1  0;
                   0  0  0  1 ];

    Tra_z = @(d) [ 1  0  0  0;
                   0  1  0  0;
                   0  0  1  d;
                   0  0  0  1 ];

    % homogeneous transform formula for the DH convention
    transform = @(alpha_i, a_i, d_i, theta_i)(Rot_z(theta_i) * Tra_z(d_i) * Tra_x(a_i) * Rot_x(alpha_i));

    
    if length(DH_parameter_table) == 0
        disp("ERROR: Forward_Kinematics: DH Table has zero entries");
        return;
    else
        disp("INFO: Forward_Kinematics: Received a table of length: "+int2str(length(DH_parameter_table)));
    end

    kinematic_equation = eye(4);

    for i=1:1:length(DH_parameter_table)
        row_i = DH_parameter_table(i, :);
        
        % extract the DH parameters
        theta_i = row_i(1);
        d_i     = row_i(2);
        a_i     = row_i(3);
        alpha_i = row_i(4);
        
        % compute the homogeneous transform
        A = transform(alpha_i, a_i, d_i, theta_i);
 
        fprintf("INFO: Forward_Kinematics: A%d%d = \n", i-1, i);
        disp(A);
        
        % update the kinematic equation
        kinematic_equation = kinematic_equation * A;

    end
    
    disp("INFO: Forward_Kinematics: Computed kinematic equation");

end
