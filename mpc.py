"""
ACC solution with MPC and IPOPT
"""
import casadi as ca, numpy as np, matplotlib.pyplot as plt
import matplotlib as mpl               # <- missing import fixed
import pathlib, os


# Parameters


dt = 0.1
tg = 1.0
tau = 0.5

u_min , u_max = -3.0, 2.0
e_max = 15.0
jerk_scale = (u_max-u_min) / dt
alpha = beta = gamma = 1/3
eps = 1e-8

sim_time = 20.0
steps = int(sim_time/dt)
u_scale = max(abs(u_min), abs(u_max))
jerk_scale = (u_max-u_min) / dt
v_lead = 5.0

N_MPC = 50

# continious model + RK integration

x = ca.MX.sym('x', 3)  # state vector [gap, velocity, acceleration]
u = ca.MX.sym('u', 1)  # control input
e, ev, a = x[0], x[1], x[2]  # unpack state vector
xdot = ca.vertcat(ev-tg*a, -a, (u-a)/tau)  # continuous dynamics

# RK4 integrator over timestep dt
F = ca.integrator('F', 'rk', {'x': x, 'p': u, 'ode': xdot}, {'t0': 0, 'tf': dt})

def step(xk, uk):
    """Simulate one timestep with RK4 integration"""
    return np.array(F(x0=xk, p=uk)['xf']).flatten()

# Single shooting optimization problem builder
def build_solver(N):
    """Build optimization problem for horizon N"""
    # Decision variables
    X = ca.MX.sym('X', 3, N+1)  # states
    U = ca.MX.sym('U', 1, N)    # controls
    P = ca.MX.sym('P', 3)       # initial state parameter
    
    # Initialize cost and constraints
    cost = 0
    g = [X[:, 0] - P]  # initial state constraint

    # Build cost and dynamics constraints over horizon
    for k in range(N):
        ek = X[0, k]  # gap error
        uk = U[0, k]  # control input
        
        # Compute jerk (zero for first timestep)
        jerk = (X[2,k]-X[2,k-1])/dt if k else 0
        
        # Add stage cost (normalized quadratic terms with smoothing)
        cost += (alpha*ca.sqrt((ek/e_max)**2+eps) +
                beta *ca.sqrt((uk/u_scale)**2+eps) +
                gamma*ca.sqrt((jerk/jerk_scale)**2+eps))
        
        # Add dynamics constraint
        g.append(X[:, k+1] - F(x0=X[:, k], p=uk)['xf'])

    # Format optimization problem
    w = ca.vertcat(ca.reshape(X,-1,1), ca.reshape(U,-1,1))
    prob = {'x':w, 'p':P, 'f':cost, 'g':ca.vertcat(*g)}
    
    # Solver options
    opts = {'ipopt.print_level':0, 'ipopt.max_iter':80,
            'ipopt.hessian_approximation':'limited-memory'}
    
    return ca.nlpsol('solver','ipopt',prob,opts), 3*(N+1), N

# Build solvers for MPC and full horizon optimization
solver_mpc, nx_mpc, nu_mpc = build_solver(N_MPC)
solver_ipo, nx_ipo, nu_ipo = build_solver(steps)

def bounds(nx, nu):
    """Generate bounds for states and controls"""
    lbx = np.concatenate([-np.inf*np.ones(nx), u_min*np.ones(nu)])
    ubx = np.concatenate([np.inf*np.ones(nx), u_max*np.ones(nu)])
    return lbx, ubx, np.zeros(nx), np.zeros(nx)

# Get bounds for both problems
lbx_m, ubx_m, lbg_m, ubg_m = bounds(nx_mpc, nu_mpc)
lbx_i, ubx_i, lbg_i, ubg_i = bounds(nx_ipo, nu_ipo)


# initial state

x0 = np.array([5.0, v_lead, 0.0])


# IPOPT benchamrk


# Solve optimization problem using IPOPT
sol_i = solver_ipo(x0=np.zeros(nx_ipo+nu_ipo), p=x0,
                   lbx=lbx_i, ubx=ubx_i, lbg=lbg_i, ubg=ubg_i)

# Extract optimal control inputs
w_ipo = sol_i['x'].full().flatten()
U_ipo = w_ipo[-nu_ipo:]

# Initialize state trajectory arrays
gap_i = [x0[0]]    # Position gap
v_i = [0.0]        # Velocity
a_i = [0.0]        # Acceleration 
u_i = []           # Control inputs
j_i = [0.0]        # Jerk

# Simulate system with optimal controls
x_current = x0.copy()
for control_input in U_ipo:
    # Store control input
    u_i.append(control_input)
    
    # Simulate one step
    x_next = step(x_current, control_input)
    
    # Calculate and store jerk
    j_i.append((x_next[2] - x_current[2])/dt)
    
    # Update states
    x_current = x_next
    
    # Store trajectory
    gap_i.append(x_current[0])
    v_i.append(v_lead - x_current[1])
    a_i.append(x_current[2])

# Convert lists to numpy arrays
gap_i, v_i, a_i, u_i, j_i = map(np.array, (gap_i, v_i, a_i, u_i, j_i))


# 5-B) Receding-horizon MPC -------------------------------------------------
time = np.arange(steps + 1) * dt

# Initialize arrays
gap_m = np.empty(steps + 1)
v_m = np.empty(steps + 1) 
a_m = np.empty(steps + 1)
u_m = np.empty(steps)
j_m = np.empty(steps + 1)

# Set initial conditions
gap_m[0], v_m[0], a_m[0], j_m[0] = x0[0], 0.0, 0.0, 0.0
w_prev = np.zeros(nx_mpc + nu_mpc)
x_meas = x0.copy()

# Run MPC loop
for k in range(steps):
    # Solve optimization problem
    sol = solver_mpc(x0=w_prev, p=x_meas,
                    lbx=lbx_m, ubx=ubx_m, lbg=lbg_m, ubg=ubg_m)
    
    # Extract optimal control and apply first input
    w_opt = sol['x'].full().flatten()
    u0 = w_opt[-nu_mpc:][0]
    u_m[k] = u0
    
    # Simulate system one step forward
    x_next = step(x_meas, u0)
    j_m[k + 1] = (x_next[2] - x_meas[2]) / dt
    x_meas = x_next
    
    # Store states
    gap_m[k + 1] = x_meas[0]
    v_m[k + 1] = v_lead - x_meas[1]
    a_m[k + 1] = x_meas[2]

    # Prepare warm start for next iteration
    Xf = w_opt[:nx_mpc].reshape(3, N_MPC + 1, order='F')
    Uf = w_opt[nx_mpc:].reshape(1, N_MPC, order='F')
    w_prev = np.concatenate((
        np.hstack((Xf[:, 1:], Xf[:, -1:])).flatten(order='F'),
        np.hstack((Uf[:, 1:], Uf[:, -1:])).flatten(order='F')
    ))

# --------------------------------------------------------------------------
# 6) Plotting
# --------------------------------------------------------------------------
mpl.rcParams.update({
    "figure.figsize": (8, 6),
    "font.size": 10,
    "axes.grid": True,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False
})

# Create subplots
fig, axs = plt.subplots(2, 2, sharex=True)
ax_e, ax_v, ax_ua, ax_j = axs.flatten()

# (a) Gap error plot
ax_e.plot(time, gap_m, lw=1.7, label=f"MPC (N={N_MPC})")
ax_e.plot(time, gap_i, ":", lw=1.7, label="IPO")
ax_e.set_ylabel("Gap error e [m]")
ax_e.set_title("(a) Gap error")
ax_e.legend()

# (b) Speed plot
ax_v.plot(time, v_m, lw=1.5, label="Follower MPC")
ax_v.plot(time, v_i, ":", lw=1.5, label="Follower IPO")
ax_v.plot(time, np.ones_like(time) * v_lead, "--", lw=1.2, label="Lead")
ax_v.set_ylabel("Speed [m/s]")
ax_v.set_title("(b) Speeds")
ax_v.legend()

# (c) Control and acceleration plot
ln1 = ax_ua.step(time[:-1], u_m, where="post", lw=1.5, label="u  MPC")
ln2 = ax_ua.step(time[:-1], u_i, where="post", lw=1.2, linestyle=":", label="u  IPO")
ax2 = ax_ua.twinx()
ln3 = ax2.plot(time, a_m, "k--", lw=1.2, label="a_i MPC")
ln4 = ax2.plot(time, a_i, ":", c="k", lw=1.2, label="a_i IPO")
ax_ua.set_ylabel(r"$u$  [m/s$^{2}$]")
ax2.set_ylabel(r"$a_i$  [m/s$^{2}$]")
ax_ua.set_title("(c) u and a_i")
ax_ua.legend(ln1 + ln2 + ln3 + ln4,
            [l.get_label() for l in ln1 + ln2 + ln3 + ln4],
            loc="upper right")

# (d) Jerk plot
ax_j.plot(time, j_m, lw=1.5, label="jerk MPC")
ax_j.plot(time, j_i, ":", lw=1.5, label="jerk IPO")
ax_j.set_ylabel("jerk  [m/s³]")
ax_j.set_title("(d) Jerk")
ax_j.set_xlabel("Time  [s]")
ax_j.legend()

fig.tight_layout()

# Save plots
outdir = pathlib.Path("plots")
outdir.mkdir(exist_ok=True)

# Save combined plot
fig.savefig(outdir / f"combined_N{N_MPC}.pdf")

# Save individual panels
fig.canvas.draw()  # Force draw for tightbbox
for ax, name in zip((ax_e, ax_v, ax_ua, ax_j), ("gap", "speed", "u_a", "jerk")):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.dpi_scale_trans.inverted()
    )
    fig.savefig(outdir / f"{name}_N{N_MPC}.png", bbox_inches=bbox)

plt.show()
