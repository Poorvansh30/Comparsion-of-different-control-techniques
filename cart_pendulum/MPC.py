import numpy as np
import cvxpy as cp
from LQR import LQRController # We reuse the linearized A and B matrices from LQR

class MPCController:
    """
    Model Predictive Control.
    Solves an optimization problem at EVERY time step to look into the future,
    predict what will happen, and choose the best sequence of moves.
    """
    def __init__(self, env, horizon=15, Q=None, R=None):
        self.N = horizon # Prediction horizon (how many steps to look ahead)
        self.dt = env.dt
        
        # Grab the continuous matrices from our LQR logic
        lqr_math = LQRController(env)
        A_cont = lqr_math.A
        B_cont = lqr_math.B
        
        # Convert Continuous to Discrete time (Forward Euler method)
        # X_new = X_old + (A * X_old + B * U) * dt
        # X_new = (I + A*dt) * X_old + (B*dt) * U
        self.A_d = np.eye(4) + A_cont * self.dt
        self.B_d = B_cont * self.dt
        
        # Cost matrices (Allow custom weights for comparison)
        if Q is not None:
            self.Q = Q
        else:
            self.Q = np.diag([1.0, 1.0, 100.0, 1.0]) 
            
        if R is not None:
            self.R = R
        else:
            self.R = np.array([[0.1]])
        
        # Max force constraint
        self.u_max = 20.0 

    def get_action(self, current_state, dt):
        """Sets up and solves the cvxpy convex optimization problem."""
        
        # Variables for states (x) and inputs (u) over the horizon
        x = cp.Variable((4, self.N + 1))
        u = cp.Variable((1, self.N))
        
        cost = 0
        constraints = []
        
        # Initial condition constraint (where are we right now?)
        constraints.append(x[:, 0] == current_state)
        
        # Loop through the prediction horizon
        for k in range(self.N):
            # Cost function: Add penalties for bad states and high forces
            cost += cp.quad_form(x[:, k], self.Q) + cp.quad_form(u[:, k], self.R)
            
            # Physics constraint: The next predicted state must obey the physics model
            constraints.append(x[:, k+1] == self.A_d @ x[:, k] + self.B_d @ u[:, k])
            
            # Actuator constraint: Motors can only push so hard
            constraints.append(cp.abs(u[:, k]) <= self.u_max)
            
        # Terminal cost (ensure it ends up in a good place)
        cost += cp.quad_form(x[:, self.N], self.Q * 10)
        
        # Formulate and solve the problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            # OSQP is a fast, reliable solver for this type of math
            prob.solve(solver=cp.OSQP, warm_start=True) 
            
            # Return the VERY FIRST control action computed for the future
            return u.value[0, 0]
        except Exception:
            # If the solver fails (e.g. pole fell too far and it's physically impossible to save)
            return 0.0