class PIDController:
    """A standard Proportional-Integral-Derivative Controller."""
    def __init__(self, kp, ki, kd, limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        
        self.integral = 0.0
        self.prev_error = 0.0

    def calculate(self, error, dt):
        # Proportional
        p_out = self.kp * error
    
        # Derivative
        derivative = (error - self.prev_error) / dt
        d_out = self.kd * derivative
        self.prev_error = error
    
        # Calculate potential output BEFORE clamping
        # We keep the integral separate to check if we are saturated
        i_out = self.ki * (self.integral + error * dt)
        raw_output = p_out + i_out + d_out
    
        # ANTI-WINDUP LOGIC:
        # Only update the integral if the system is NOT saturated.
        # If the output is clamped, we stop adding to the integral.
        if self.limit is None or abs(raw_output) <= self.limit:
                self.integral += error * dt
    
        # Now calculate final output
        final_output = p_out + (self.ki * self.integral) + d_out
    
        # Final Clamp
        if self.limit is not None:
            final_output = max(-self.limit, min(self.limit, final_output))
        
        return final_output

class CartPolePID:
    """
    To balance a cart-pole, you need TWO PID controllers.
    One to keep the angle at 0, and one to keep the cart position at 0.
    They fight each other slightly, which is why manual tuning is hard!
    """
    def __init__(self, kp_ang, kd_ang, kp_pos, kd_pos, ki_ang=0.0):
        # We usually ignore the integral term for this specific problem 
        # to prevent lag, acting as a PD controller.
        self.pid_angle = PIDController(kp_ang, ki_ang, kd_ang)
        self.pid_position = PIDController(kp_pos, 0.0, kd_pos)
 
    def get_action(self, state, dt):
        x, x_dot, theta, theta_dot = state
        
        # We want theta to be 0. Error is (0 - theta)
        angle_error = 0.0 - theta
        # We want position to be 0. Error is (0 - x)
        pos_error = 0.0 - x
        
        force_angle = self.pid_angle.calculate(angle_error, dt)
        force_pos = self.pid_position.calculate(pos_error, dt)
        
        # The total force applied to the cart
        # (Angle control is strictly more important than position)
        return force_angle + force_pos

