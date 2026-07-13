import numpy as np
import matplotlib.pyplot as plt
import time
from physics_env import CartPolePlant
from PID import CartPolePID
from LQR import LQRController
from MPC import MPCController
import PID_ml
def run_simulation(controller, name, env, steps=300):
    """Runs a single simulation and collects data for the report."""
    # Start with an initial disturbance: angle = 0.2 rad (~11 degrees)
    state = env.reset([0.0, 0.0, 0.2, 0.0])
    
    history_t = []
    history_theta = []
    history_x = []
    history_force = []
    
    start_time = time.time()
    for i in range(steps):
        t = i * env.dt
        force = controller.get_action(state, env.dt)
        state, done = env.step(force)
        
        history_t.append(t)
        history_x.append(state[0])
        history_theta.append(state[2])
        history_force.append(force)
        
        if done:
            break
            
    exec_time = time.time() - start_time
    return history_t, history_theta, history_x, history_force, name, exec_time

def calculate_metrics(time, theta, force):
    """Calculates report metrics: Overshoot and Control Effort."""
    # Peak angle experienced after time 0
    peak_angle = max(np.abs(theta))
    # Overshoot relative to the starting angle (0.2 rad)
    overshoot = (peak_angle - 0.2) / 0.2 * 100 if peak_angle > 0.2 else 0.0
    
    # Control effort (integral of squared force)
    control_effort = np.sum(np.array(force)**2)
    
    # Settling time (time to stay within 0.01 rad of 0)
    settled_idx = -1
    for i in range(len(theta)-1, -1, -1):
        if abs(theta[i]) > 0.01:
            settled_idx = i
            break
    settling_time = time[settled_idx] if settled_idx > 0 else time[-1]
            
    return overshoot, settling_time, control_effort

def main():
    env = CartPolePlant(dt=0.02)
    
    print("Initializing Controllers and calculating gains...")
    
    # Measure Setup/Synthesis Time for PID
    t0 = time.time()
    manual_pid = CartPolePID(kp_ang=-200.0, kd_ang=-40.0, kp_pos=-5.0, kd_pos=-10.0)
    setup_pid = time.time() - t0
    
    # Measure Setup/Synthesis Time for LQR
    t0 = time.time()
    # You can easily change Q and R here to test different LQR performance!
    # e.g., LQRController(env, Q=np.diag([10, 1, 100, 1]), R=np.array([[1.0]]))
    lqr = LQRController(env)
    setup_lqr = time.time() - t0
    
    # Measure Setup/Synthesis Time for MPC
    t0 = time.time()
    # You can easily change Q and R here to test different MPC performance!
    mpc = MPCController(env, horizon = 25)
    setup_mpc = time.time() - t0
    
    # Measure Setup/Synthesis Time for ML-PID
    print("Training ML PID... (This takes a few seconds)")
    t0 = time.time()
    ml_gains = PID_ml.run_ml_training()
    ml_pid = CartPolePID(*ml_gains)
    setup_ml = time.time() - t0
    
    controllers = [
        (manual_pid, "Manual PID", setup_pid),
        (ml_pid, "ML-Optimized PID", setup_ml),
        (lqr, "LQR", setup_lqr),
        (mpc, "MPC", setup_mpc)
    ]
    
    results = []
    
    print("\nRunning Simulations...")
    for ctrl, name, setup_time in controllers:
        res = run_simulation(ctrl, name, env)
        # Pack all results including the setup time
        results.append((*res, setup_time))
        
    # Generate the OVERALL Comparison Plot
    plt.figure("Comparative Overview", figsize=(12, 12))
    
    print("\n" + "="*95)
    print(f"{'Controller':<18} | {'Synthesis Time':<14} | {'Execution Time':<14} | {'Overshoot':<10} | {'Settling':<9} | {'Effort'}")
    print("-" * 95)
    
    for t, theta, x, force, name, exec_time, setup_time in results:
        # Plot Angle
        plt.subplot(3, 1, 1)
        plt.plot(t, np.array(theta) * (180/np.pi), label=name, linewidth=2)
        
        # Plot Position
        plt.subplot(3, 1, 2)
        plt.plot(t, x, label=name, linewidth=2)
        
        # Plot Force
        plt.subplot(3, 1, 3)
        plt.plot(t, force, label=name, linewidth=1.5, alpha=0.8)
        
        # Print Metrics
        os, st, ce = calculate_metrics(t, theta, force)
        print(f"{name:18} | {setup_time:12.6f} s | {exec_time:12.6f} s | {os:8.1f} % | {st:6.2f} s | {ce:8.1f}")
        
    print("="*95 + "\n")
        
    plt.subplot(3, 1, 1)
    plt.title("Comparative: Pendulum Angle Stabilization")
    plt.ylabel("Angle (degrees)")
    plt.axhline(0, color='black', linestyle='--')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.title("Comparative: Cart Position")
    plt.ylabel("Position (meters)")
    plt.axhline(0, color='black', linestyle='--')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.title("Comparative: Control Effort (Motor Force)")
    plt.ylabel("Force (N)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Generate INDIVIDUAL Plots for each controller
    for t, theta, x, force, name, exec_time, setup_time in results:
        plt.figure(f"Individual: {name}", figsize=(10, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(t, np.array(theta) * (180/np.pi), color='blue', linewidth=2)
        plt.title(f"{name} - Pendulum Angle")
        plt.ylabel("Angle (deg)")
        plt.axhline(0, color='black', linestyle='--')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(t, x, color='orange', linewidth=2)
        plt.title(f"{name} - Cart Position")
        plt.ylabel("Position (m)")
        plt.axhline(0, color='black', linestyle='--')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(t, force, color='green', linewidth=2)
        plt.title(f"{name} - Control Effort")
        plt.ylabel("Force (N)")
        plt.xlabel("Time (s)")
        plt.grid(True)
        
        plt.tight_layout()
        
    plt.show()

if __name__ == "__main__":
    main()