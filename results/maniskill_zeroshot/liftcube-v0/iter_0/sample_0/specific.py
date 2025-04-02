import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get positions
    ee_pos = self.tcp.pose.p
    obj_pos = self.obj.pose.p
    
    # Check grasp
    grasp_success = self.agent.check_grasp(self.obj)
    
    # Milestone 1: Reach the cube
    distance_to_cube = np.linalg.norm(ee_pos - obj_pos)
    reward += max(0, 1 - distance_to_cube / 0.1)  # Reward for being close to the cube
    
    # Milestone 2: Grasp the cube
    if grasp_success:
        reward += 1.0  # Reward for successful grasp
    else:
        reward -= 0.1  # Penalty for not grasping
    
    # Milestone 3: Lift the cube by 0.2 meters
    if grasp_success:
        target_height = obj_pos[2] + 0.2
        current_height = obj_pos[2]
        height_diff = target_height - current_height
        reward += max(0, 1 - abs(height_diff) / 0.2)  # Reward for lifting the cube
    
    # Milestone 4: Keep the cube static after lifting
    if grasp_success and check_actor_static(self.obj):
        reward += 0.5  # Reward for keeping the cube static
    
    # Penalize large actions for smoothness
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward