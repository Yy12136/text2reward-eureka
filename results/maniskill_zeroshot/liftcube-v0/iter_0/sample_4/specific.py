import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get positions
    ee_pos = self.tcp.pose.p
    cubeA_pos = self.obj.pose.p
    
    # Check grasp
    grasp_success = self.agent.check_grasp(self.obj)
    
    # Step 1: Move end-effector close to cube A
    distance_to_cube = np.linalg.norm(ee_pos - cubeA_pos)
    reward += 1.0 / (1.0 + distance_to_cube)  # Inverse distance reward
    
    # Step 2: Grasp cube A
    if grasp_success:
        reward += 1.0  # Bonus for successful grasp
    else:
        # Penalize if the end-effector is close but not grasping
        if distance_to_cube < 0.05:
            reward -= 0.5
    
    # Step 3: Lift cube A by 0.2 meter
    if grasp_success:
        target_height = cubeA_pos[2] + 0.2
        current_height = cubeA_pos[2]
        height_diff = target_height - current_height
        reward += 1.0 / (1.0 + abs(height_diff))  # Inverse height difference reward
        
        # Bonus for reaching the target height
        if abs(height_diff) < 0.01:
            reward += 1.0
    
    # Step 4: Ensure cube A is static after lifting
    if grasp_success and check_actor_static(self.obj):
        reward += 1.0  # Bonus for keeping the cube static
    
    # Penalize large joint velocities to encourage smooth motion
    joint_vel = np.linalg.norm(self.agent.robot.get_qvel()[:-2])
    reward -= 0.1 * joint_vel
    
    return reward