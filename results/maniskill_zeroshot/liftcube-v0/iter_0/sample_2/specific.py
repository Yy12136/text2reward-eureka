import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get positions
    ee_pos = self.tcp.pose.p
    obj_pos = self.obj.pose.p
    
    # Check grasp
    grasp_success = self.agent.check_grasp(self.obj)
    
    # Milestone 1: Move end-effector close to cube A
    dist_to_cube = np.linalg.norm(ee_pos - obj_pos)
    reward += max(0, 1.0 - 5.0 * dist_to_cube)  # Reward decreases with distance
    
    # Milestone 2: Grasp cube A
    if grasp_success:
        reward += 2.0  # Large reward for successful grasp
    
    # Milestone 3: Lift cube A by 0.2 meters
    if grasp_success:
        target_height = obj_pos[2] + 0.2
        height_diff = target_height - obj_pos[2]
        reward += max(0, 1.0 - 10.0 * abs(height_diff))  # Reward decreases with height difference
    
    # Milestone 4: Ensure cube A is static after lifting
    if grasp_success and check_actor_static(self.obj):
        reward += 1.0  # Reward for stability
    
    # Penalize large actions to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward