import numpy as np
from scipy.optimize import differential_evolution
from physics_env import CartPolePlant
from PID import CartPolePID

def evaluate_pid_gains(gains):
    """
    The 'Fitness Function' for our Machine Learning algorithm.
    Takes a set of guessed PID gains, runs a simulation, and scores them.
    Lower score = Better.
    """
    kp_ang, kd_ang, kp_pos, kd_pos = gains
    
    # Initialize physics and our PID controller with the guessed gains
    env = CartPolePlant()
    controller = CartPolePID(kp_ang, kd_ang, kp_pos, kd_pos)
    
    # Start slightly tipped over to force it to react
    state = env.reset([0.0, 0.0, 0.2, 0.0])
    
    total_cost = 0.0
    steps = 250 # 5 seconds of simulation
    
    for _ in range(steps):
        force = controller.get_action(state, env.dt)
        state, done = env.step(force)
        
        x, _, theta, _ = state
        
        # The Cost calculation: 
        # We heavily penalize the angle deviating from 0.
        # We lightly penalize the cart moving away from the center.
        # We lightly penalize using too much force.
        step_cost = (theta ** 2) * 100.0 + (x ** 2) * 1.0 + (force ** 2) * 0.01
        total_cost += step_cost
        
        if done:
            # The agent completely failed (pole fell over). Huge penalty!
            total_cost += 10000.0 * (steps - _) # Penalize failing early
            break
            
    return total_cost

def run_ml_training():
    """
    Uses Differential Evolution (a genetic ML algorithm) to find optimal gains.
    """
    print("Starting ML Training (Differential Evolution)...")
    
    # Define the search space bounds for the 4 gains: 
    # [kp_ang, kd_ang, kp_pos, kd_pos]
    # We force them to be negative because our error math was (0 - state)
    bounds = [(-300, 0), (-100, 0), (-50, 0), (-50, 0)]
    
    # Run the genetic algorithm
    # popsize: How many 'agents' per generation
    # maxiter: How many generations to breed
    result = differential_evolution(
        evaluate_pid_gains, 
        bounds, 
        popsize=15, 
        maxiter=30, 
        disp=True # Prints progress to the console
    )
    
    best_gains = result.x
    print("\n--- ML Training Complete ---")
    print(f"Optimal Kp_angle: {best_gains[0]:.2f}")
    print(f"Optimal Kd_angle: {best_gains[1]:.2f}")
    print(f"Optimal Kp_pos:   {best_gains[2]:.2f}")
    print(f"Optimal Kd_pos:   {best_gains[3]:.2f}")
    
    return best_gains

if __name__ == "__main__":
    # If you run this file directly, it will train and print the gains.
    run_ml_training()