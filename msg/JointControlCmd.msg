# Joint control command message for PD control with feedforward torque
# All arrays should have the same size (number of actuated joints)

std_msgs/Header header

# Desired joint positions [rad]
float64[] position

# Desired joint velocities [rad/s]
float64[] velocity

# Feedforward torque [Nm]
float64[] torque

# Position gain (Kp)
float64[] kp

# Velocity/Damping gain (Kd)
float64[] kd

