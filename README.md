# Comparsion-of-different-control-techniques
I made a program in python which simulates the motion of inverted pendulum on the cart, and tries to maintain the upright position of the pendulum by applying different control laws like PID, LQR, and MPC. 
To run the program, you can just run the main file, also you can observe the behaviour of the pendulum and control force using the graphs plotted. 
You can manually change the PID gains for manual pid tuning controller and also change weight matrices for the LQR controller in LQR.py file.
You can also make changes to the MPC.py file, where you can change the number of horizons, that the program looks forward to, maximum actuator force, and the cost function, which is used to optimise the algorithm. 
