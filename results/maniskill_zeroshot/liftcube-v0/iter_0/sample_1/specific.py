import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get positions
    ee_pos = self.tcp.pose.p
    cubeA_pos = self.obj.pose.p
    
    # Check grasp
    grasp_success = self.agent.check_grasp(self.obj)
    
    # Milestone 1: Move end-effector close to cube A
    distance_to_cubeA = np.linalg.norm(ee_pos - cubeA_pos)
    reward += max(0, 1 - distance_to_cubeA / 0.1)  # Reward based on proximity
    
    # Milestone 2: Grasp cube A
    if grasp_success:
        reward += 1.0  # Reward for successful grasp
    
    # Milestone 3: Lift cube A by 0.2 meter
    if grasp_success:
        target_height = cubeA_pos[2] + 0.2
        current_height = cubeA_pos[2]
        height_diff = abs(target_height - current_height)
        reward += max(0, 1 - height_diff / 0.2)  # Reward based on height difference
    
    # Milestone 4: Ensure cube A is static after lifting
    if grasp_success and check_actor_static(self.obj):
        reward += 1.0  # Reward for keeping cube A static
    
    # Penalize large actions to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward