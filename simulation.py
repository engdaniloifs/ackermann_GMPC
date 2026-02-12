

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
import geometric_mpc
from GMPC_Tracking_Control_main.utils.enum_class import TrajType, ControllerType
from ref_traj_generator import TrajGenerator
import nonlinear_mpc
import feedback_linearization




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




controller_type = 'NMPC'  # 'GMPC', 'NMPC', 'FBLINEARIZATION'
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
                                'linear_vel': 0.25,
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
                                'linear_vel': 0.75,
                                'angular_vel': 0.1,  # don't change this
                                'radius': 1.0,
                                'nTraj': 600,
                                'controller_type': controller_type}}
    
traj_gen = TrajGenerator(traj_config)
ref_state, ref_control, dt = traj_gen.get_traj()

if controller_type == 'GMPC':
    controller = geometric_mpc.GeometricMPC(traj_config)
    Q = np.array([0.04, 0.04, 0.04])
    R = 0.3
    N = 10
    controller.setup_solver(Q, R, N)
if controller_type == 'NMPC':
    controller = nonlinear_mpc.NonlinearMPC(traj_config,model_config={}, dt= dt)
    Q = np.array([300, 300, 600, 600])
    R = np.array([500, 0.8])
    N = 10
    controller.setup_solver(Q, R, N)
if controller_type == 'FBLINEARIZATION':
    controller = feedback_linearization.FBLinearizationController(Kp = np.array([20, 20, 20, 20]))



ref_state, ref_control, dt = traj_gen.get_traj()
euclidean_error = np.zeros(ref_state.shape[1])
theta_error = np.zeros(ref_state.shape[1])

L = 0.256

if controller_type == 'GMPC':
    nTraj = ref_state.shape[1]

    x = np.zeros((5, nTraj))
    u = np.zeros((2, nTraj))

    desired_u = np.zeros((2, nTraj))

    x[0,0] = -2.5
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
        
        
        error_dist[i] = np.hypot(x[0,i] - ref_state[0,i], x[1,i] - ref_state[1,i])
        
        
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

    x[0,0] = -2.5
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
plt.show()

# %%
# MAKE AN ANIMATION

# Create and save the animation
ani = vehicle.animate(
    x,
    T,
    0,
    0,
    True,
    "../agv-book/gifs/ch3/ackermann_kinematic.gif",
)

# Show the movie to the screen
plt.show()

# # Show animation in HTML output if you are using IPython or Jupyter notebooks
# plt.rc('animation', html='jshtml')
# display(ani)
# plt.close()
