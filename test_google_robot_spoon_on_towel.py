import os
# Attempt headless operation by setting DISPLAY env var to empty
# This needs to be done BEFORE SAPIEN/Vulkan attempts to initialize a display
os.environ["DISPLAY"] = ""

import simpler_env
import numpy as np
import mediapy as media
# Ensure get_image_from_maniskill2_obs_dict is correctly imported.
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sys

# Ensure ManiSkill2_real2sim and its parent directory are in sys.path
# This is to handle potential import issues in various execution environments.
# Path to the directory that CONTAINS the 'mani_skill2_real2sim' package
# This path is /app/ManiSkill2_real2sim, which contains the actual package 'mani_skill2_real2sim'
ms2_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ManiSkill2_real2sim'))
if ms2_repo_path not in sys.path:
    sys.path.insert(0, ms2_repo_path)

# Add /app itself to sys.path to allow 'import simpler_env'
# This is important if the script is not run with /app as the current working directory.
app_path = os.path.abspath(os.path.dirname(__file__))
if app_path not in sys.path:
     sys.path.insert(0, app_path)


def main():
    env = None  # Initialize env to None for the finally block
    try:
        print("Initializing environment: google_robot_spoon_on_towel")
        # simpler_env.make already sets obs_mode='rgbd' and prepackaged_config=True
        env = simpler_env.make("google_robot_spoon_on_towel")
        print("Environment initialized.")

        print("Resetting environment...")
        obs, reset_info = env.reset()
        print("Environment reset.")

        instruction = env.get_language_instruction()
        print(f"Language Instruction: {instruction}")
        print(f"Reset Info: {reset_info}")

        images = []
        # Add initial image
        print("Capturing initial image...")
        img = get_image_from_maniskill2_obs_dict(env, obs)
        if img is not None:
            images.append(img)
            print("Initial image captured.")
        else:
            print("Warning: Failed to capture initial image.")
        
        num_steps = 20 
        print(f"Starting random action loop for {num_steps} steps...")
        for i in range(num_steps):
            # Generate a random action: 6D for EE delta pose (normalized -1 to 1), 1D for gripper.
            # The control mode is "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
            # This means the gripper action is a delta for the target joint position.
            action = np.random.uniform(low=-0.1, high=0.1, size=7).astype(np.float32) # Arm delta
            action[6] = np.random.uniform(low=-0.01, high=0.01).astype(np.float32) # Gripper delta

            print(f"\nStep {i+1}/{num_steps}, Action: {action}")
            # The environment step typically returns: obs, reward, terminated, truncated, info
            # ManiSkill2 environments often return success in `info['success']`.
            # The gymnasium standard is obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 'success' is often in the info dict for ManiSkill tasks
            success = info.get('success', False) 

            print(f"Timestep: {i+1}, Success: {success}, Terminated: {terminated}, Truncated: {truncated}, Reward: {reward}")
            # For clarity, only print info if needed, as it can be verbose
            # print(f"Info for step {i+1}: {info}")

            print(f"Capturing image for step {i+1}...")
            img_step = get_image_from_maniskill2_obs_dict(env, obs)
            if img_step is not None:
                images.append(img_step)
                print(f"Image for step {i+1} captured.")
            else:
                print(f"Warning: Failed to capture image for step {i+1}.")

            if success or terminated or truncated:
                print(f"Episode finished early at step {i+1}. Success: {success}, Terminated: {terminated}, Truncated: {truncated}")
                break
        
        if images and len(images) > 0 : # Ensure images list is not empty
            video_path = "test_google_robot_spoon_on_towel.mp4"
            print(f"\nSaving video with {len(images)} frames to {video_path}...")
            media.write_video(video_path, images, fps=5)
            print("Test script finished. Video saved to test_google_robot_spoon_on_towel.mp4")
        else:
            print("\nNo images were collected to save a video.")
            print("Test script finished. No video saved.")


    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            print("Closing environment...")
            env.close()
            print("Environment closed.")

if __name__ == "__main__":
    main()
