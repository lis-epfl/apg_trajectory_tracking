import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=3)

from neural_control.utils.straight import Hover, Straight
from neural_control.utils.circle import Circle
from neural_control.utils.polynomial import Polynomial

# the input to generate a reference trajectory is a drone state of the form
# position, attitude, velocity
drone_state = np.random.rand(9) - .5

# for any of the trajectories, you need to specify the following parameters:
arguments = {
    "dt": 0.05,  # time per step (in s)
    "horizon":
        10,  # (in steps --> we will get the next 10 states in the reference
    "max_drone_dist": 0.25  # (in m)
}

# HOVER
print("\n------- Hover --------------")
hov_traj = Hover(drone_state, **arguments)
# in my hover reference, the drone should stay at one fixed position, which is
# the same as its initial position
print(
    "Target position", hov_traj.target_pos, ", Drone position", drone_state[:3]
)

# Get the next reference states (number = horizon)
# input the drone state and its acceleration
# (here simply assumed to be equal to the attitude)
# say, the drone has now been displaced a bit from the target state
current_drone_state = drone_state.copy()
current_drone_state[:3] += (np.random.rand(3) * .1)
drone_acceleration = current_drone_state[3:6]
# the reference traj is then computed with the minimum snap trajectory code
hover_reference = hov_traj.get_ref_traj(drone_state, drone_acceleration)
print("Hover reference: (position (3), velocity (3), acceleration(3))")
print(hover_reference)

# STRAIGHT
print("\n----- Straight ----------")
straight_traj = Straight(drone_state, **arguments)
collect_references = []
for _ in range(10):
    straight_reference = straight_traj.get_ref_traj(
        drone_state, drone_acceleration
    )
    collect_references.append(straight_reference)
    # assume perfect controller -->
    # put the drone at the last point of the reference
    drone_state[:3] = straight_reference[-1, :3]  # pos
    drone_state[6:9] = straight_reference[-1, 3:6]  # vel
positions_together = np.vstack(collect_references)[:, :3]
plt.plot(positions_together)
plt.title("Reference positions in straight trajectory")
plt.show()
print(
    "Note: The reference is not perfectly straight because the minimum snap\
    trajectory takes the drone slowly towards the straight trajectory,\
    considering velocity and acceleration"
)

# CIRCLE
print("\n-------- Circle -----------")
# using 2D circles in either x-y plane, or x-z plane, or y-z plane
arguments["radius"] = 2  # in m
arguments["plane"] = [0, 1]
# make circle
circle_traj = Circle(drone_state, **arguments)
collect_references = []
for _ in range(200):
    circle_reference = circle_traj.get_ref_traj(drone_state, drone_state[3:6])
    collect_references.append(drone_state[:3].copy())
    # assume perfect controller that achieves to move to the next ref state
    drone_state[:3] = circle_reference[0, :3]  # pos
    drone_state[6:9] = circle_reference[0, 3:6]  # vel
    drone_state[3:6] = circle_reference[0, 6:]  # acc
positions_together = np.array(collect_references)
plt.scatter(positions_together[:, 0], positions_together[:, 1])
plt.title("Reference positions in circle trajectory")
plt.show()

# POLYNOMIAL
print("--------- Polynomial ---------------")
poly_traj = Polynomial(drone_state, **arguments)
# generating a random polynomial in 3D when initialized
lines = plt.plot(poly_traj.reference[:, :3])
plt.title("Polynomial reference")
plt.legend(iter(lines), ["x", "y", "z"])
plt.show()
print(
    "Note: in the beginning and in the end the drone is supposed to hover.\
    The number of hover steps can be controlled with the hover argument."
)
# In this case, we cannot simply project the drone onto the reference traj
# Instead, we keep track of a target index and return the minimum snap
# trajectory to reach the target index
print(
    f"\nCurrent target index {poly_traj.target_ind} (trajectory\
    length: {len(poly_traj.reference)})... run tracking"
)

collect_references = []
for _ in range(500):
    poly_reference = poly_traj.get_ref_traj(drone_state, drone_state[3:6])
    collect_references.append(drone_state[:3].copy())
    # assume imperfect controller that achieves to move close to the next state
    drone_state[[0, 1, 2, 6, 7, 8, 3, 4, 5]] = poly_reference[0]
    drone_state[:3] += (np.random.rand(3) * .03)

positions_together = np.array(collect_references)
lines = plt.plot(positions_together[:, :3])
plt.legend(iter(lines), ["x", "y", "z"])
plt.title(
    "Positions when following polynomial trajectory (imperfect controller)"
)
plt.show()
print("\nTarget index after tracking 500 steps:", poly_traj.target_ind)
print(
    "Note: If the state is not perfectly tracked, the drone is slower than the\
    reference (the target index is not increased in every step)"
)
