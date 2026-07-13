import numpy as np
from scipy.integrate import solve_ivp

class CartPolePlant:
    """
    The non-linear physics engine for the Cart-Pole system.
    This simulates the real-world physics using the user's derived 
    equations of motion for a point-mass pendulum.
    """
    def __init__(self, mass_cart=1.0, mass_pole=0.1, length=0.5, gravity=9.81, dt=0.02):
        # Physical parameters of the system
        self.M = mass_cart    # m_c in equations
        self.m = mass_pole    # m_p in equations
        self.l = length       # L in equations
        self.g = gravity      # g in equations
        self.dt = dt          # Time step for the simulation
        
        # State vector: [x (pos), x_dot (vel), theta (angle), theta_dot (ang vel)]
        self.state = np.zeros(4)

    def reset(self, initial_state=None):
        """Resets the system to an initial state."""
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=float)
        else:
            self.state = np.zeros(4)
        return self.state

    def _equations_of_motion(self, t, state, force):
        """
        Calculates the state derivatives (velocity and acceleration) 
        using the explicitly derived Newtonian equations.
        """
        x, x_dot, theta, theta_dot = state
        
        # Pre-compute sines and cosines
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Cart Acceleration (x_ddot)
        numerator_x = force + self.m * sin_theta * (-self.g * cos_theta + self.l * (theta_dot ** 2))
        denominator_x = self.M + self.m * (sin_theta ** 2)
        x_ddot = numerator_x / denominator_x
        
        # Pendulum Angular Acceleration (theta_ddot)
        theta_ddot = (self.g * sin_theta - x_ddot * cos_theta) / self.l
        
        # [velocity, acceleration, ang_velocity, ang_acceleration]
        return [x_dot, x_ddot, theta_dot, theta_ddot]

    def step(self, force):
        """
        Steps the simulation forward by dt seconds using the RK45 numerical solver.
        """
        # We integrate from time 0 to dt
        result = solve_ivp(
            fun=lambda t, y: self._equations_of_motion(t, y, force),
            t_span=(0, self.dt),
            y0=self.state,
            method='RK45'
        )
        
        # Extract the final state at the end of the time step
        self.state = result.y[:, -1]
        
        # Determine if the pole fell over (failed state). 
        # Pi/2 radians is 90 degrees (completely flat).
        done = bool(abs(self.state[2]) > np.pi / 2) 
        
        return self.state, done