

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
import geometric_mpc
import geometric_mpc_ackermann 
from GMPC_Tracking_Control_main.utils.enum_class import TrajType, ControllerType
from ref_traj_generator import TrajGenerator
import nonlinear_mpc
import feedback_linearization
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D




# Ackermann vehicle parameters
#controller
def wrap_to_pi(a):
    return (a+np.pi)% (2*np.pi) -np.pi

def rk4_step(x, u, dt, f):
    k1 = f(x, u)
    k2 = f(x + 0.5*dt*k1, u)
    k3 = f(x + 0.5*dt*k2, u)
    k4 = f(x + dt*k3, u)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def step_discrete_actuators(x, u, dt):
    L = 0.256
    tau_phi = 0.16
    
    tau_v = 0.1
    if abs(x[4])<0.05:
        tau_v = 2*tau_v
    phi_max = 0.5

    x_pos, y_pos, theta, phi, v = x
    v_cmd, omega_cmd = u

    
    # safe mapping speed for yaw-rate -> steering

    phi_des = np.arctan2(L * omega_cmd, v)
    phi_des = np.clip(phi_des, -phi_max, phi_max)

    a_phi = 1.0 - np.exp(-dt / tau_phi)
    a_v   = 1.0 - np.exp(-dt / tau_v)
    phi_next = phi + a_phi * (phi_des - phi)
    
    if v_cmd*v<0:
        v_next   = v   + a_v   * (v_cmd  - v)    
        if v_next*v<0:
            v_next =0.0
    else:
        v_next   = v   + a_v   * (v_cmd  - v)
    if abs(v_next)<0.05:
        v_next=0.0

    return phi_next, v_next


def ackermann_kinematic_model(x, u):
    """
    Ackermann kinematic model
    x: state vector [x, y, theta]
    u: control vector [v, phi]
    returns: state derivative vector [dx/dt, dy/dt, dtheta/dt]
    """
    L = 0.256

    x_pos, y_pos, theta = x
    phi,v= u

    
    
    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dtheta = v * np.tan(phi) / L

    return np.array([dx, dy, dtheta])

def step(x, u, dt):
    # update actuator states (still states!)
    dt_actuator = 0.020
    number_of_steps = int(dt / dt_actuator)
    curr_x = x.copy()
    curr_u = x[3:5].copy()
    for _ in range(number_of_steps):
        
        # integrate pose using updated actuators
        curr_x[0:3] = rk4_step(curr_x[0:3], curr_u, dt_actuator, ackermann_kinematic_model)
        
        curr_x[2] = wrap_to_pi(curr_x[2])
        curr_u = step_discrete_actuators(curr_x, u, dt_actuator) + np.array([0.02*np.random.uniform(-1,1), 0.05*np.random.uniform(-1,1)])

        curr_x[3] = curr_u[0]
        curr_x[4] = curr_u[1]
    return curr_x




controller_type = 'GMPC_ACKERMANN'  # 'GMPC', 'NMPC', 'FBLINEARIZATION'
if controller_type == 'GMPC':
    init_state = np.array([0, 0, 0])
    traj_config = {'type': 'CIRCLE_LEADER_FOLLOWER',
                    'param': {'start_state': np.array([-2.5, -1.5, 0]),
                                'middle_state': np.array([0, -1.5, 0]),
                                'dt': 0.05,
                                'linear_vel': 0.25,
                                'angular_vel': 0.1,  # don't change this
                                'radius': 1.0,
                                'nTraj': 600,
                                'controller_type': controller_type}}
elif controller_type == 'NMPC':
    init_state = np.array([0, 0, 0, 0])
    traj_config = {'type': 'CIRCLE_LEADER_FOLLOWER',
                    'param': {'start_state': np.array([-2.5, -1.5, 0, 0]),
                                'middle_state': np.array([0, -1.5, 0, 0]),
                                'dt': 0.05,
                                'linear_vel': 0.75,
                                'angular_vel': 0.1,  # don't change this
                                'radius': 1.0,
                                'nTraj': 600,
                                'controller_type': controller_type}}
elif controller_type == 'FBLINEARIZATION':
    init_state = np.array([0, 0, 0, 0])
    traj_config = {'type': 'CIRCLE_LEADER_FOLLOWER',
                    'param': {'start_state': np.array([-2.5, -1.5, 0, 0]),
                                'middle_state': np.array([0, -1.5, 0, 0]),
                                'dt': 0.05,
                                'linear_vel': 0.25,
                                'angular_vel': 0.1,  # don't change this
                                'radius': 1.0,
                                'nTraj': 600,
                                'controller_type': controller_type}}
elif controller_type == 'GMPC_ACKERMANN':
    init_state = np.array([0, 0, 0])
    traj_config = {'type': 'CIRCLE_LEADER_FOLLOWER',
                    'param': {'start_state': np.array([-2.5, -1.5, 0]),
                                'middle_state': np.array([0, -1.5, 0]),
                                'dt': 0.05,
                                'linear_vel': 0.25,
                                'angular_vel': 0.1,  # don't change this
                                'radius': 1.0,
                                'nTraj': 600,
                                'controller_type': 'GMPC'}}

traj_gen = TrajGenerator(traj_config)
ref_state, ref_control, dt = traj_gen.get_traj()

if controller_type == 'GMPC':
    controller = geometric_mpc.GeometricMPC(traj_config)
    Q = np.array([600, 600, 150])
    R = 1500
    N = 13
    controller.setup_solver(Q, R, N)
if controller_type == 'NMPC':
    controller = nonlinear_mpc.NonlinearMPC(traj_config,model_config={}, dt= dt)
    Q = np.array([600, 600, 150, 50])
    R = np.array([1500, 0.05])
    N = 5
    controller.setup_solver(Q, R, N)
if controller_type == 'FBLINEARIZATION':
    controller = feedback_linearization.FBLinearizationController(Kp = np.array([1, 4.5, 16, 6]))
if controller_type == 'GMPC_ACKERMANN':
    controller = geometric_mpc_ackermann.GeometricMPC_ackermann(traj_config)
    Q = np.array([600, 600, 150])
    R = np.array([1500, 0.05])
    N = 13
    controller.setup_solver(Q, R, N)


ref_state, ref_control, dt = traj_gen.get_traj()
euclidean_error = np.zeros(ref_state.shape[1])
theta_error = np.zeros(ref_state.shape[1])

L = 0.256

if controller_type == 'GMPC':
    nTraj = ref_state.shape[1]

    x = np.zeros((5, nTraj))
    u = np.zeros((2, nTraj))

    desired_u = np.zeros((2, nTraj))

    x[0,0] = -2.9
    x[1,0] = -1.5
    x[2,0] = 0
    x[3,0] = 0
    x[4,0] = 0
    t = 0

    v_min = -1.5
    v_max= 1.5
    w_min = -3.0
    w_max = 3.0

    controller.set_control_bound(v_min, v_max, w_min, w_max)

    error_dist = np.zeros(nTraj)


    curr_state = np.array([x[0,0], x[1,0], x[2,0]])


    curr_actuators = np.array([x[4,0], x[3,0]])
    tau_v = 0.05

    for i in range(1,nTraj):
        x[:3,i] = rk4_step(curr_state, curr_actuators, dt, ackermann_kinematic_model)
        #x[:,i] = step(x[:,i-1], desired_u[:,i-1], dt)
        x[2,i] = wrap_to_pi(x[2,i])
        curr_state = np.array([x[0,i], x[1,i], x[2,i]])
        euclidean_error[i] = np.hypot(x[0,i] - ref_state[0,i], x[1,i] - ref_state[1,i])
        theta_error[i] = wrap_to_pi(x[2,i] - ref_state[2,i])
        
        
        
        desired_u[:,i]= controller.solve(curr_state, t)

        phi = np.arctan2(L * desired_u[1,i], x[4,i-1])
        phi_des = np.clip(phi, -0.5, 0.5)
        a_phi = 1.0 - np.exp(-dt / 0.16)
        a_v   = 1.0 - np.exp(-dt / tau_v)
        v_next   = x[4,i-1] + a_v   * (desired_u[0,i]  - x[4,i-1])  
        
        phi_next = x[3,i-1] + a_phi * (phi_des - x[3,i-1])
        x[3,i] = phi_next
        x[4,i] = v_next 
        curr_actuators = np.array([x[3,i], x[4,i]])
        t += dt
        
        print("step",i,"out of ",nTraj)
elif controller_type == 'NMPC': 
    nTraj = ref_state.shape[1]

    x = np.zeros((5, nTraj))
    u = np.zeros((2, nTraj))

    desired_u = np.zeros((2, nTraj))

    x[0,0] = -2.9
    x[1,0] = -1.5
    x[2,0] = 0
    x[3,0] = 0
    x[4,0] = 0

    t = 0

    v_min = -1.75
    v_max= 1.75


    controller.set_control_bound(v_min, v_max)


    curr_state = np.array([x[0,0], x[1,0], x[2,0]])


    curr_actuators = np.array([x[4,0], u[1,0]])
    tau_v = 0.05
    for i in range(1,nTraj):
        x[:3,i] = rk4_step(curr_state, curr_actuators, dt, ackermann_kinematic_model)
        #x[:,i] = step(x[:,i-1], desired_u[:,i-1], dt)
        x[2,i] = wrap_to_pi(x[2,i])
        curr_state = np.array([x[0,i], x[1,i], x[2,i]])
        euclidean_error[i] = np.hypot(x[0,i] - ref_state[0,i], x[1,i] - ref_state[1,i])
        theta_error[i] = wrap_to_pi(x[2,i] - ref_state[2,i])
        curr_state_1 = np.array([x[0,i], x[1,i], x[2,i], x[3,i-1]])

        desired_u[:,i]= controller.solve(curr_state_1, t)
        steering_ratio = desired_u[1,i]
        v_cmd = desired_u[0,i]
        phi_des = x[3,i-1] + steering_ratio*dt
        phi_des = np.clip(phi_des, -0.5, 0.5)
        
        
        a_phi = 1.0 - np.exp(-dt / 0.16)
        a_v   = 1.0 - np.exp(-dt / tau_v)
        
        v_next   = x[4,i-1] + a_v   * (desired_u[0,i]  - x[4,i-1])  
        
        phi_next = x[3,i-1] + a_phi * (phi_des - x[3,i-1])
        x[3,i] = phi_next
        x[4,i] = v_next
        curr_actuators = np.array([x[3,i], x[4,i]])
        t += dt
        print("step",i,"out of ",nTraj)
elif controller_type == 'FBLINEARIZATION':
    nTraj = ref_state.shape[1]

    x = np.zeros((5, nTraj))
    u = np.zeros((2, nTraj))

    desired_u = np.zeros((2, nTraj))

    x[0,0] = -2.9
    x[1,0] = -1.5
    x[2,0] = 0
    x[3,0] = 0
    x[4,0] = 0

    t = 0

    v_min = -1.5
    v_max= 1.5


    curr_state = np.array([x[0,0], x[1,0], x[2,0]])


    curr_actuators = np.array([x[4,0], u[1,0]])
    tau_v = 0.05
    for i in range(1,nTraj):
        x[:3,i] = rk4_step(curr_state, curr_actuators, dt, ackermann_kinematic_model)
        #x[:,i] = step(x[:,i-1], desired_u[:,i-1], dt)
        x[2,i] = wrap_to_pi(x[2,i])
        curr_state = np.array([x[0,i], x[1,i], x[2,i]])
        euclidean_error[i] = np.hypot(x[0,i] - ref_state[0,i], x[1,i] - ref_state[1,i])
        theta_error[i] = wrap_to_pi(x[2,i] - ref_state[2,i])
        curr_state_1 = np.array([x[0,i], x[1,i], x[2,i], x[3,i-1]])

        desired_u[:,i]= controller.feedback_control(curr_state_1, ref_state[:,i-1], ref_control[:,i-1])
        steering_ratio = desired_u[1,i]
        v_cmd = max(min(desired_u[0,i], v_max), v_min)
        phi_des = x[3,i-1] + steering_ratio*dt
        phi_des = np.clip(phi_des, -0.5, 0.5)
        
        
        a_phi = 1.0 - np.exp(-dt / 0.16)
        a_v   = 1.0 - np.exp(-dt / tau_v)
        
        v_next   = x[4,i-1] + a_v   * (v_cmd  - x[4,i-1])  
        
        phi_next = x[3,i-1] + a_phi * (phi_des - x[3,i-1])
        x[3,i] = phi_next
        x[4,i] = v_next
        curr_actuators = np.array([x[3,i], x[4,i]])
        t += dt
        print("step",i,"out of ",nTraj)
if controller_type == 'GMPC_ACKERMANN':
    nTraj = ref_state.shape[1]

    x = np.zeros((5, nTraj))
    u = np.zeros((2, nTraj))

    desired_u = np.zeros((2, nTraj))

    x[0,0] = -2.9
    x[1,0] = -1.5
    x[2,0] = 0
    x[3,0] = 0
    x[4,0] = 0
    t = 0

    v_min = -1.75
    v_max= 1.75
    w_min = -2.34
    w_max = 2.34

    controller.set_control_bound(v_min, v_max, w_min, w_max)

    error_dist = np.zeros(nTraj)


    curr_state = np.array([x[0,0], x[1,0], x[2,0]])


    curr_actuators = np.array([x[4,0], x[3,0]])
    tau_v = 0.05

    for i in range(1,nTraj):
        x[:3,i] = rk4_step(curr_state, curr_actuators, dt, ackermann_kinematic_model)
        #x[:,i] = step(x[:,i-1], desired_u[:,i-1], dt)
        x[2,i] = wrap_to_pi(x[2,i])
        curr_state = np.array([x[0,i], x[1,i], x[2,i]])
        euclidean_error[i] = np.hypot(x[0,i] - ref_state[0,i], x[1,i] - ref_state[1,i])
        theta_error[i] = wrap_to_pi(x[2,i] - ref_state[2,i])
        
        
        
        desired_u[:,i]= controller.solve(curr_state, t)

        phi = np.arctan2(L * desired_u[1,i], 1)
        phi_des = np.clip(phi, -0.5, 0.5)
        a_phi = 1.0 - np.exp(-dt / 0.16)
        a_v   = 1.0 - np.exp(-dt / tau_v)
        v_next   = x[4,i-1] + a_v   * (desired_u[0,i]  - x[4,i-1])  
        
        phi_next = x[3,i-1] + a_phi * (phi_des - x[3,i-1])
        x[3,i] = phi_next
        x[4,i] = v_next 
        curr_actuators = np.array([x[3,i], x[4,i]])
        t += dt
        
        print("step",i,"out of ",nTraj)



t = np.arange(0.0,nTraj*dt, dt)
# Plot the states as a function of time
fig1 = plt.figure(1)
fig1.set_figheight(6.4)
ax1a = plt.subplot(311)
plt.plot(t, x[0, :])
plt.plot(t, ref_state[0,:],'r--')
plt.grid(color="0.95")
plt.ylabel(r"$x$ [m]")
plt.setp(ax1a, xticklabels=[])
ax1b = plt.subplot(312)
plt.plot(t, x[1, :])
plt.plot(t, ref_state[1,:],'r--')
plt.grid(color="0.95")
plt.ylabel(r"$y$ [m]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(313)
plt.plot(t, x[2, :] * 180.0 / np.pi)
plt.plot(t, ref_state[2,:]* 180.0 / np.pi, 'r--')
plt.grid(color="0.95")
plt.ylabel(r"$\theta$ [deg]")
plt.xlabel(r"$t$ [s]")
plt.legend()
# Save the plot
#plt.savefig("../agv-book/figs/ch3/ackermann_kinematic_fig1.pdf")

# Plot the position of the vehicle in the plane
fig2 = plt.figure(2)
plt.plot(x[0, :], x[1, :])
plt.plot(ref_state[0, :], ref_state[1, :], 'b--')
plt.axis("equal")


fig3 = plt.figure(3)
ax1b = plt.subplot(211)
plt.plot(t, x[4, :])
plt.grid(color="0.95")
plt.ylabel(r"$v$ [m/s]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(212)
plt.plot(t, x[3, :] * 180.0 / np.pi)
plt.grid(color="0.95")
plt.ylabel(r"$\phi$ [deg]")
plt.xlabel(r"$t$ [s]")
plt.legend()
plt.plot()

fig4 = plt.figure(4)
ax4a = plt.subplot(211)
plt.plot(t, euclidean_error)
plt.grid(color="0.95")
plt.ylabel(r"Euclidean error [m]")
plt.setp(ax4a, xticklabels=[])
ax4b = plt.subplot(212)
plt.plot(t, theta_error * 180.0 / np.pi)
plt.grid(color="0.95")
plt.ylabel(r"Heading error [deg]")
plt.xlabel(r"$t$ [s]")
plt.legend()
plt.plot()
# Save the plot
#plt.savefig("../agv-book/figs/ch3/ackermann_kinematic_fig2.pdf")

# Show all the plots to the screen




# x: [x, y, theta, phi, ...]
# ref_state: [x_ref, y_ref, ...]
# dt defined

# ---------------------------
# Geometry (tune)
# ---------------------------
L = 0.256          # wheelbase [m]
track = 0.18       # track width [m]
rear_overhang = 0.05
front_overhang = 0.07

body_length = L + rear_overhang + front_overhang
body_width  = track + 0.08

wheel_len = 0.06
wheel_w   = 0.03

# Wheel centers in BODY frame (x forward, y left)
rear_axle_x = 0.0
front_axle_x = L
yL, yR = +track/2, -track/2

p_rl = np.array([rear_axle_x,  yL])
p_rr = np.array([rear_axle_x,  yR])
p_fl = np.array([front_axle_x, yL])
p_fr = np.array([front_axle_x, yR])

def R(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s],
                     [s,  c]])

# ---------------------------
# Figure
# ---------------------------
fig, ax = plt.subplots()
ax.grid(color="0.95")
ax.set_aspect("equal", adjustable="box")
ax.set_title("Trajectory (car + wheels + steering)")

ax.plot(ref_state[0, :], ref_state[1, :], "b--", label="ref")
traj_line, = ax.plot([], [], label="traj", linewidth=2)

# Axis limits
xmin, xmax = x[0, :].min(), x[0, :].max()
ymin, ymax = x[1, :].min(), x[1, :].max()
mx = 0.1 * (xmax - xmin + 1e-9)
my = 0.1 * (ymax - ymin + 1e-9)
ax.set_xlim(xmin - mx, xmax + mx)
ax.set_ylim(ymin - my, ymax + my)

# ---------------------------
# Patches (defined in LOCAL frames)
# ---------------------------

# Body is defined in BODY frame with origin at rear axle center
body = Rectangle((-rear_overhang, -body_width/2), body_length, body_width,
                 fill=False, linewidth=2)

# Wheels defined in their OWN local frame centered at origin (0,0)
def make_wheel():
    return Rectangle((-wheel_len/2, -wheel_w/2), wheel_len, wheel_w,
                     fill=True, alpha=0.8)

w_rl = make_wheel()
w_rr = make_wheel()
w_fl = make_wheel()
w_fr = make_wheel()

for p in [body, w_rl, w_rr, w_fl, w_fr]:
    ax.add_patch(p)

ax.legend(loc="best")

# ---------------------------
# Helpers: set transforms
# ---------------------------
def set_body_pose(xw, yw, th):
    tr = Affine2D().rotate(th).translate(xw, yw) + ax.transData
    body.set_transform(tr)

def set_wheel_pose(wheel_patch, center_world_xy, wheel_angle):
    cx, cy = center_world_xy
    tr = Affine2D().rotate(wheel_angle).translate(cx, cy) + ax.transData
    wheel_patch.set_transform(tr)

# ---------------------------
# Animation
# ---------------------------
stride = 1
frames = range(0, x.shape[1], stride)

def init():
    traj_line.set_data([], [])

    xw, yw, th, phi = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
    set_body_pose(xw, yw, th)

    pos = np.array([xw, yw])
    Rth = R(th)
    c_rl = pos + Rth @ p_rl
    c_rr = pos + Rth @ p_rr
    c_fl = pos + Rth @ p_fl
    c_fr = pos + Rth @ p_fr

    set_wheel_pose(w_rl, c_rl, th)
    set_wheel_pose(w_rr, c_rr, th)
    set_wheel_pose(w_fl, c_fl, th + phi)
    set_wheel_pose(w_fr, c_fr, th + phi)

    return traj_line, body, w_rl, w_rr, w_fl, w_fr

def update(k):
    traj_line.set_data(x[0, :k+1], x[1, :k+1])

    xw, yw = x[0, k], x[1, k]
    th = x[2, k]
    phi = x[3, k]

    set_body_pose(xw, yw, th)

    pos = np.array([xw, yw])
    Rth = R(th)
    c_rl = pos + Rth @ p_rl
    c_rr = pos + Rth @ p_rr
    c_fl = pos + Rth @ p_fl
    c_fr = pos + Rth @ p_fr

    # rear wheels: angle = theta
    set_wheel_pose(w_rl, c_rl, th)
    set_wheel_pose(w_rr, c_rr, th)

    # front wheels: angle = theta + phi (steering)
    set_wheel_pose(w_fl, c_fl, th + phi)
    set_wheel_pose(w_fr, c_fr, th + phi)

    return traj_line, body, w_rl, w_rr, w_fl, w_fr

ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                    interval=dt * 1000 * stride, blit=True)

plt.show()

# Save (optional):
# ani.save("traj_car_wheels.mp4", fps=int(round(1/dt/stride)))