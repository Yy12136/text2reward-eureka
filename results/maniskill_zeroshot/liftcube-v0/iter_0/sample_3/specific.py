import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get positions
    ee_pos = self.tcp.pose.p
    obj_pos = self.obj.pose.p
    
    # Check grasp
    grasp_success = self.agent.check_grasp(self.obj)
    
    # Step 1: Approach the cube
    distance_to_cube = np.linalg.norm(ee_pos - obj_pos)
    reward += max(0, 1.0 - distance_to_cube) * 0.5  # Encourage getting closer to the cube
    
    # Step 2: Grasp the cube
    if grasp_success:
        reward += 1.0  # Reward for successful grasp
    else:
        reward -= 0.1  # Penalize for not grasping
    
    # Step 3: Lift the cube
    if grasp_success:
        desired_height = obj_pos[2] + 0.2  # Desired height is 0.2 meters above the current position
        current_height = obj_pos[2]
        height_diff = abs(desired_height - current_height)
        reward += max(0, 1.0 - height_diff) * 0.5  # Encourage lifting to the desired height
    
    # Step 4: Ensure the cube is static after lifting
    if grasp_success and check_actor_static(self.obj):
        reward += 1.0  # Reward for keeping the cube static after lifting
    
    # Penalize for large joint velocities to encourage smooth motion
    joint_velocities = np.linalg.norm(self.agent.robot.get_qvel()[:-2])
    reward -= 0.01 * joint_velocities
    
    return reward