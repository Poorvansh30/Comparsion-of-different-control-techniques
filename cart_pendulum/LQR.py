import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController:
        """
        Linear Quadratic Regulator.
        This calculates mathematically optimal gains by linearizing the physics
        around the upright position (theta = 0, x = 0).
        """
        def __init__(self, env, Q=None, R=None):
            # We extract physics parameters from the environment
            M = env.M
            m = env.m
            l = env.l
            g = env.g
            
            # 1. LINEARIZE THE SYSTEM (A and B matrices)
            # These are derived by taking the Jacobian of the equations of motion
            # evaluated at theta = 0.
            
            denominator = M * l
            
            self.A = np.array([
                [0, 1, 0, 0],
                [0, 0, -(m * g) / M, 0],
                [0, 0, 0, 1],
                [0, 0, ((M + m) * g) / denominator, 0]
            ])
            
            self.B = np.array([
                [0],
                [1 / M],
                [0],
                [-1 / denominator]
            ])
            
            # 2. DEFINE THE COST MATRICES (Q and R)
            # Allow user to pass custom Q and R to compare performance!
            if Q is not None:
                self.Q = Q
            else:
                self.Q = np.diag([1.0, 1.0, 100.0, 10.0])
                
            if R is not None:
                self.R = R
            else:
                self.R = np.array([[0.1]])
                
            # 3. SOLVE THE ALGEBRAIC RICCATI EQUATION
            # This solves for P in the equation: A^T P + P A - P B R^-1 B^T P + Q = 0
            P = solve_continuous_are(self.A, self.B, self.Q, self.R)
            
            # 4. CALCULATE THE OPTIMAL GAIN MATRIX (K)
            # K = R^-1 * B^T * P
            self.K = np.linalg.inv(self.R).dot(self.B.T).dot(P)

        def get_action(self, state, dt):
            """
            LQR is a full-state feedback controller: u = -Kx
            """
            # Ensure state is a column vector
            x = np.array(state).reshape(4, 1)
            
            # Calculate optimal force
            u = -self.K.dot(x)
            
            return u[0, 0] # Return as a scalar