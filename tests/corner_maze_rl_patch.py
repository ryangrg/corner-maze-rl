from __future__ import annotations

from datetime import datetime

from typing import Optional, Dict, Any, Tuple

from collections import deque

import numpy as np
import random
import matplotlib.pyplot as plt

import torch as th

from constants_patch import * 

from gymnasium import spaces
from gymnasium.core import ObsType

from minigrid.core.constants import COLORS, COLOR_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Ball, Floor
from minigrid.core.actions import Actions
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_rect                      
from minigrid.wrappers import ImgObsWrapper

from functools import partial
import time
import os
from screeninfo import get_monitors

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

import torch  # Import torch
import torch.nn as nn  # Import nn from PyTorch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import gymnasium as gym

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

# Rotated coordinate helper tables
DEFAULT_AGENT_START_POS = rotate_point((1, 1))
DEFAULT_AGENT_START_DIR = rotate_dir(0)

CORNER_LEFT_TURN_FIXES = {
    rotate_point((2, 2)): (rotate_point((3, 2)), rotate_dir(0)),
    rotate_point((10, 2)): (rotate_point((10, 3)), rotate_dir(1)),
    rotate_point((10, 10)): (rotate_point((9, 10)), rotate_dir(2)),
    rotate_point((2, 10)): (rotate_point((2, 9)), rotate_dir(3)),
}

WELL_EXIT_FORWARD_FIXES = {
    rotate_point((1, 1)): (rotate_point((2, 2)), rotate_dir(1)),
    rotate_point((11, 1)): (rotate_point((10, 2)), rotate_dir(2)),
    rotate_point((11, 11)): (rotate_point((10, 10)), rotate_dir(3)),
    rotate_point((1, 11)): (rotate_point((2, 10)), rotate_dir(0)),
}

WELL_ENTRY_PICKUP_FIXES = {
    rotate_point((10, 2)): (rotate_point((11, 1)), rotate_dir(0)),
    rotate_point((10, 10)): (rotate_point((11, 11)), rotate_dir(1)),
    rotate_point((2, 10)): (rotate_point((1, 11)), rotate_dir(2)),
    rotate_point((2, 2)): (rotate_point((1, 1)), rotate_dir(3)),
}

TURN_ONE_NS_MAP = {
    rotate_point((6, 5)): 1,
    rotate_point((6, 7)): 0,
    rotate_point((5, 6)): 1,
    rotate_point((7, 6)): 0,
}

TURN_ONE_EW_MAP = {
    rotate_point((6, 5)): 0,
    rotate_point((6, 7)): 1,
    rotate_point((5, 6)): 0,
    rotate_point((7, 6)): 1,
}

TURN_TWO_SET_A = {rotate_point(point) for point in [(7, 2), (5, 10), (2, 7), (10, 5)]}
TURN_TWO_SET_B = {rotate_point(point) for point in [(5, 2), (7, 10), (2, 5), (10, 7)]}

# Place the minigrid in the alternate monitor
# Set the monitor index to choose the target monitor
monitor_index = 1  # Adjust as needed for the monitor (e.g., 0, 1, 2...)

# Get the position of the chosen monitor
monitors = get_monitors()
if monitor_index < len(monitors):
    monitor = monitors[monitor_index]
    x, y = monitor.x, monitor.y
else:
    monitor = monitors[0]
    x, y = monitor.x, monitor.y

# Set the SDL environment variable for window position
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"

# Check if MPS is available and set the device
use_gpu = 1
if use_gpu:
    device = torch.device("mps")
    print("MPS is available and set as the device.")
else:
    device = torch.device("cpu")
    print("MPS is not available. Using CPU instead.")

# REGION ######################### BEGIN ENVIRONMENT CODE ###############################
# Here se define the environment model after the real world corner maze task
# The environment is a corner maze with a set of walls, cues, rewards, and triggers.
# The agent can navigate through the maze, and learn to achieve specific goals based on cues and rewards.
# This a dynamic environment where the layout can change based on the phase of the session.
# In regards to the environment, session and trial are used in the context of behavioral neuroscience experiments.
# A session is a series of trials, and a trial is a single instance of the task.
# Extend colors for customs objects

COLORS["cue_on_rgb"] = np.array([255, 0, 255])
COLORS["cue_off_rgb"] = np.array([25, 0, 255])
COLORS["chasm_rgb"] = np.array([0, 0, 255])
COLORS["wall_rgb"] = np.array([0, 255, 0])
COLORS["black"] = np.array([0, 0, 0])
COLOR_TO_IDX["cue_on_rgb"] = 6
COLOR_TO_IDX["cue_off_rgb"] = 7
COLOR_TO_IDX["chasm_rgb"] = 8
COLOR_TO_IDX["wall_rgb"] = 9
COLOR_TO_IDX["black"] = 10   

# Extended grid objects to build corner maze environment
class Chasm(Wall):
    def __init__(self, color='chasm_rgb'):
        super().__init__(color)

    def see_behind(self):
        # Can the agent see through this cell?
        return True
    
class Barrier(Wall):
    def __init__(self, color='wall_rgb'):
        super().__init__(color)

    def see_behind(self):
        # Can the agent see through this cell?
        return True

class RewardWell(Ball):
    def __init__(self, color='black', visble=False):
        super().__init__(color)
        self.visible = visble

    def can_toggle(self):
        return True
    
    def can_overlap(self):
        return True
    
    def render(self, img):
        if self.visible:
            c = COLORS[self.color]
            fill_coords(img, point_in_rect(0, 1, 0, 1), c)

class EmptyWell(Ball):
    def __init__(self, color='black', visble=False):
        super().__init__(color)
        self.visible = visble

    def can_overlap(self):
        return True
    
    def can_toggle(self):
        return True

    def render(self, img):
        if self.visible:
            c = COLORS[self.color]
            fill_coords(img, point_in_rect(0, 1, 0, 1), c)

class Trigger(Floor):
    def __init__(self, color='black', visble=False, trigger_type=None):
        super().__init__(color)
        self.visible = visble
        self.color = color
        self.trigger_type = trigger_type
        # Validate and set the type
        if trigger_type not in ('A', 'B', 'S', None):
            raise ValueError("trigger_type must be 'A', 'B', 'S', or None")
    
    def get_trigger_type(self):
        return self.trigger_type

    def render(self, img):
        if self.visible:
            c = COLORS[self.color]
            fill_coords(img, point_in_rect(0, 1, 0, 1), c)

# Extend MiniGridEnv to build corner maze environment
class CornerMazeEnv(MiniGridEnv):
    # Set fps here for monitoring agent performance visually
    metadata = {"render_modes": ["human"], "render_fps": RENDER_FPS}
    def __init__(
        self,
        size=13,
        agent_start_pos=DEFAULT_AGENT_START_POS,
        agent_start_dir=DEFAULT_AGENT_START_DIR,
        max_steps: int | None = None,
        # Added initialization variables
        session_type: str | None = None,
        agent_cue_goal_orientation: str | None = None,
        start_goal_location: str | None = None,
        run_mode: int = 0,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 4 * size**2

        # Set added initialization variables
        self.session_type = session_type
        self.agent_cue_goal_orientation = agent_cue_goal_orientation
        self.start_goal_location = start_goal_location
        self.run_mode = run_mode

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        # Define the observation space
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255, shape=(AGENT_VIEW_SIZE, AGENT_VIEW_SIZE, 3), dtype=np.uint8
            ),
            "direction": spaces.Discrete(4),
            "mission": mission_space,
        })

        # Range of possible rewards
        self.reward_range = (-1, 1)

        # Shared class variables
        self.init_variables()
    
    # Initialize class variables
    def init_variables(self):
        """
        Initializes various variables and configurations for the maze environment.
        This method sets up the initial state of the maze, including the maze state array, 
        different layout configurations, and agent point of view settings. It dynamically 
        builds layouts for different trial configurations based on start arms, cues, and goals. 
        It also sets up inter-trial interval (ITI) configurations and generates sequences for 
        grid configurations and starting positions.
        Key initializations include:
        - `self.maze_state_array`: A list representing the initial state of the maze.
        - `self.layouts`: A dictionary to store different maze layouts.
        - `self.maze_config_trl_list`: A 3D list to store trial layouts indexed by start arms, cues, and goals.
        - `self.maze_config_iti_list`: A 2D list to store ITI configurations indexed by start arms and ITI types.
        - `self.grid_configuration_sequence`: A sequence of grid configurations for the session.
        - `self.start_pose`: The starting position of the agent.
        - `self.agent_pov_pos`: The agent's point of view position.
        - `self.visual_mask`: A mask representing the agent's visual field.
        - `self.wall_ledge_mask`: A mask representing the wall ledge.
        This method is essential for setting up the maze environment and ensuring that the 
        agent interacts with the maze according to the specified configurations.
        """
        # state_type: [0] 0 = base, 1 = exposure, 2 = trial, 3 = iti, 
        # maze_state_array: [1-16] = barriers, [17-20] = cues, [21-24] = wells, 
        # [25-36] = trigger zones. reference spread sheet for more details
        self.maze_state_array = [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        
        # Dictionary to store trl layouts
        self.layouts = {}

        # Initial layout configuration, no barriers, no cues, no rewards, no triggers
        # Layout dynamic naming: layout_phase_{start_arm}_{cue}_{goal}
        self.layouts['layout_x_x_xx'] = [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        
        # exposure configurations
        self.layouts['layout_exp_x_x_ne'] = [1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        self.layouts['layout_exp_x_x_se'] = [1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0, 0,0,0,0, 0,1,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        self.layouts['layout_exp_x_x_sw'] = [1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0, 0,0,0,0, 0,0,1,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        self.layouts['layout_exp_x_x_nw'] = [1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]

        # Define dimensions for start arm, cue, and goal
        start_arms = ['n', 'e', 's', 'w']
        cues = ['n', 'e', 's', 'w', 'x']
        goals = ['ne', 'se', 'sw', 'nw']

        # Dynamically build layout variables for all trial configurations
        # This follows maze layouts as defined in the 2S2C behavioral task
        # variable naming: layout_trl_{start_arm}_{cue}_{goal}
        base_trl_layouts = {
            'n' : [2, 1,0,1, 0,0,0, 1,0,1, 0,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0],
            'e' : [2, 0,0,0, 1,0,1, 0,0,0, 1,0,1, 0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0],
            's' : [2, 1,0,1, 0,0,0, 1,0,1, 0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0],
            'w' : [2, 0,0,0, 1,0,1, 0,0,0, 1,0,1, 0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        }
        for start_arm in start_arms:
            for cue in cues:
                for goal in goals:
                    variable_name = f'layout_trl_{start_arm}_{cue}_{goal}'
                    layout = base_trl_layouts[start_arm].copy()
                    layout[17] = 1 if cue == 'n' else 0
                    layout[18] = 1 if cue == 'e' else 0
                    layout[19] = 1 if cue == 's' else 0
                    layout[20] = 1 if cue == 'w' else 0
                    layout[21] = 1 if goal == 'ne' else 0
                    layout[22] = 1 if goal == 'se' else 0
                    layout[23] = 1 if goal == 'sw' else 0
                    layout[24] = 1 if goal == 'nw' else 0

                    self.layouts[variable_name] = layout

        # ITI Configurations: location of start arm is stated after iti
        # the meaning of the goal location is different here it indicates the type of ITI
        # such that the maze is configured to lead to the next start arm location while including that well location.
        # when the goal is xx it means two wells are present to enter but the rat must go to the far side of the maze to get to the
        # start arm. In all ITI configurations the well is in the empty state.
        self.layouts['layout_iti_n_x_xx'] = [3, 0,0,0, 0,1,0, 0,1,0, 0,1,0, 1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 2,0, 0,0, 0,1, 0,0,0,0]
        self.layouts['layout_iti_n_x_nw'] = [3, 0,0,1, 0,1,0, 0,1,0, 1,1,0, 1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 3,0,0,0]
        self.layouts['layout_iti_n_x_ne'] = [3, 1,0,0, 0,1,1, 0,1,0, 0,1,0, 1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 3,0,0,0]
        self.layouts['layout_iti_e_x_xx'] = [3, 0,1,0, 0,0,0, 0,1,0, 0,1,0, 0,1,0,0, 0,0,0,0, 0,0,0,0, 0,1, 0,0, 2,0, 0,0, 0,0,0,0]
        self.layouts['layout_iti_e_x_ne'] = [3, 1,1,0, 0,0,1, 0,1,0, 0,1,0, 0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,3,0,0]
        self.layouts['layout_iti_e_x_se'] = [3, 0,1,0, 1,0,0, 0,1,1, 0,1,0, 0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,3,0,0]
        self.layouts['layout_iti_s_x_xx'] = [3, 0,1,0, 0,1,0, 0,0,0, 0,1,0, 0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,1, 0,0, 2,0, 0,0,0,0]
        self.layouts['layout_iti_s_x_se'] = [3, 0,1,0, 1,1,0, 0,0,1, 0,1,0, 0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,3,0]
        self.layouts['layout_iti_s_x_sw'] = [3, 0,1,0, 0,1,0, 1,0,0, 0,1,1, 0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,3,0]
        self.layouts['layout_iti_w_x_xx'] = [3, 0,1,0, 0,1,0, 0,1,0, 0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,0, 2,0, 0,0, 0,1, 0,0, 0,0,0,0]
        self.layouts['layout_iti_w_x_sw'] = [3, 0,1,0, 0,1,0, 1,1,0, 0,0,1, 0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,3]
        self.layouts['layout_iti_w_x_nw'] = [3, 0,1,1, 0,1,0, 0,1,0, 1,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,3]

        # convert layout values to tuples
        self.layouts.update({k: tuple(v) for k, v in self.layouts.items()})
        self.layout_name_lookup = {v: k for k, v in self.layouts.items()}

        # configure lists for pulling layouts during session sequence construction
        # Initialize a 3D list to store the layouts for index accessing with goals, cues, and start_arms
        self.maze_config_trl_list = [[[None for _ in goals] for _ in cues] for _ in start_arms]

        # Create 3d Matrix of maze trial configurations for building session layout order
        for i, startarm in enumerate(start_arms):
            for j, cue in enumerate(cues):
                for k, goal in enumerate(goals):
                    layout_name = f'layout_trl_{startarm}_{cue}_{goal}'
                    self.maze_config_trl_list[i][j][k] = self.layouts.get(layout_name)  # Fetch the layout or None if not found

        # Create 2D Matrix of iti configurations for building session layout order
        self.maze_config_iti_list = [[None for _ in range(3)] for _ in range(4)]
        self.maze_config_iti_list[0][0] = self.layouts.get('layout_iti_n_x_xx')
        self.maze_config_iti_list[0][1] = self.layouts.get('layout_iti_n_x_nw')
        self.maze_config_iti_list[0][2] = self.layouts.get('layout_iti_n_x_ne')
        self.maze_config_iti_list[1][0] = self.layouts.get('layout_iti_e_x_xx')
        self.maze_config_iti_list[1][1] = self.layouts.get('layout_iti_e_x_ne')
        self.maze_config_iti_list[1][2] = self.layouts.get('layout_iti_e_x_se')
        self.maze_config_iti_list[2][0] = self.layouts.get('layout_iti_s_x_xx')
        self.maze_config_iti_list[2][1] = self.layouts.get('layout_iti_s_x_se')
        self.maze_config_iti_list[2][2] = self.layouts.get('layout_iti_s_x_sw')
        self.maze_config_iti_list[3][0] = self.layouts.get('layout_iti_w_x_xx')
        self.maze_config_iti_list[3][1] = self.layouts.get('layout_iti_w_x_sw')
        self.maze_config_iti_list[3][2] = self.layouts.get('layout_iti_w_x_nw')

        

        # Agent POV variables
        self.agent_pov_pos = ((AGENT_VIEW_SIZE // 2), (AGENT_VIEW_SIZE - 1) - AGENT_VIEW_BEHIND)
        # construct visual mask
        visual_exclusion = []  
        for i in range(AGENT_VIEW_BEHIND):
            if i == 0:
                visual_exclusion.append(((((AGENT_VIEW_SIZE) - AGENT_VIEW_BEHIND)+ i),AGENT_VIEW_SIZE // 2))
            else:
                for j in range(i+1):
                    if j == 0:
                        visual_exclusion.append(((((AGENT_VIEW_SIZE) - AGENT_VIEW_BEHIND) + i), AGENT_VIEW_SIZE // 2))
                    else:
                        visual_exclusion.append(((((AGENT_VIEW_SIZE) - AGENT_VIEW_BEHIND) + i), AGENT_VIEW_SIZE // 2 - j))
                        visual_exclusion.append(((((AGENT_VIEW_SIZE) - AGENT_VIEW_BEHIND) + i), AGENT_VIEW_SIZE // 2 + j))
        self.visual_mask = np.ones((AGENT_VIEW_SIZE, AGENT_VIEW_SIZE), dtype=bool)
        for cell in visual_exclusion:
            self.visual_mask[cell[0], cell[1]] = False
        self.visual_mask = self.expand_matrix(self.visual_mask, VIEW_TILE_SIZE)
        # construct wall ledge mask
        wall_ledge_inclusion = []
        wall_ledge_inclusion.append((AGENT_VIEW_SIZE - AGENT_VIEW_BEHIND - 2, AGENT_VIEW_SIZE // 2))
        wall_ledge_inclusion.extend([(AGENT_VIEW_SIZE - AGENT_VIEW_BEHIND - 1, AGENT_VIEW_SIZE // 2 - 1),
                                    (AGENT_VIEW_SIZE - AGENT_VIEW_BEHIND - 1, AGENT_VIEW_SIZE // 2),
                                    (AGENT_VIEW_SIZE - AGENT_VIEW_BEHIND - 1, AGENT_VIEW_SIZE // 2 + 1)])
        wall_ledge_inclusion.append((AGENT_VIEW_SIZE - AGENT_VIEW_BEHIND, AGENT_VIEW_SIZE // 2))
        self.wall_ledge_mask = np.ones((AGENT_VIEW_SIZE, AGENT_VIEW_SIZE, 2), dtype=bool)
        for cell in wall_ledge_inclusion:
            self.wall_ledge_mask[cell[0], cell[1], :] = False
        self.wall_ledge_mask = self.expand_matrix(self.wall_ledge_mask, VIEW_TILE_SIZE)
        
        # Initialize action space 
        # only use forward, left, right, and pickup: pickup is modify to be a well entering action
        self.action_space = spaces.Discrete(4)
        
        # action space and masking control/conditional variables
        self.last_action = None
        self.last_pose  = [None, None, None]
        self.key_actions = 0
        self.trajectory = []
        self.trajectories = []
        self.grid_configuration_len = None
        
        # scoring and session trial variables
        self.turn_score = [None, None]
        self.trial_score = None
        self.session_reward = 0
        self.session_num_trials = None
        self.trial_count = None
        self.episode_trial_scores = []
        self.episode_turn_scores = []
        self.episode_scores = []
        self.episode_rewards = []
        self.phase_step_count = None
        self.phase_punishment_scr = None #subtracts from reward if in phase too long
        self.episode_terminated = None
        self.episode_truncated = None
        self.in_place_punishment_scr = 0
        self.in_loc_count = 0 #if the same arm for more than 9 moves give penalty
        self.session_phase = None # 0: trial, 1: iti_proximal, 2: iti_distal

        # Dataframe stuff
        self.episode = 0
        self.trial_training_session_columns = ['episode', 'turn_scores', 'trial_scores',
                                               'trial_num_steps', 'trial_well_visits', 
                                               'episode_score', 'trajectory']
        self.episode_data = pd.DataFrame(columns=self.trial_training_session_columns)

        # Position variables
        self.fwd_pos = None
        self.fwd_cell = None
        self.cur_cell = None

        # Temp single trial session variables
        self.pseudo_session_score = deque(maxlen=ACQUISITION_SESSION_TRIALS)
        [self.pseudo_session_score.append(0) for _ in range(ACQUISITION_SESSION_TRIALS)]

    def expand_matrix(self, original_matrix, scale_factor):
        """
        Expand a low-resolution matrix into a higher-resolution matrix by repeating
        each element into a (scale_factor x scale_factor) block.

        Parameters
        - original_matrix: np.ndarray
            A 2D boolean/numeric matrix with shape (H, W) or a 3D matrix with shape
            (H, W, C) where C is the number of channels (e.g., 2 for masks or 3 for RGB).
        - scale_factor: int
            The integer factor by which to scale each matrix axis. Each element at
            position (i, j) in the original matrix will expand to the block
            [i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor]
            in the returned matrix.

        Returns
        - expanded_matrix: np.ndarray
            A new array with shape (H*scale_factor, W*scale_factor) for 2D inputs or
            (H*scale_factor, W*scale_factor, C) for 3D inputs. The dtype of the
            returned array matches the dtype of `original_matrix`.

                Notes
                - This implementation uses explicit Python loops for clarity and to match
                    the original behavior. For large matrices or performance-critical
                    paths, consider using `np.kron(original_matrix, np.ones((scale_factor, scale_factor)))`
                    which performs the same expansion much faster using vectorized operations.

                Effect on observations
                - The primary use of this function in the environment is to convert
                    low-resolution, cell-aligned masks (shape: AGENT_VIEW_SIZE x AGENT_VIEW_SIZE
                    or AGENT_VIEW_SIZE x AGENT_VIEW_SIZE x C) into pixel-aligned masks that
                    match the image returned by `grid.render(tile_size=VIEW_TILE_SIZE)`.
                - Typical masks in `init_variables()`:
                        - `self.visual_mask` (2D boolean): which grid cells the agent can see.
                            After expansion, this is applied to the red channel of the rendered
                            observation to zero-out cue pixels outside the agent's visual field.
                        - `self.wall_ledge_mask` (3D boolean with C==2): per-cell inclusion for
                            two separate channel masks. After expansion, the two channels are
                            applied to the green and blue channels of the rendered image to
                            selectively hide/show walls and ledges.
                - After expansion, the expected spatial shape is
                    `(AGENT_VIEW_SIZE * VIEW_TILE_SIZE, AGENT_VIEW_SIZE * VIEW_TILE_SIZE)`
                    (or with channels for 3D masks). It's good practice to assert this
                    when debugging:
                        `assert expanded.shape[:2] == (AGENT_VIEW_SIZE * VIEW_TILE_SIZE,
                                                                                         AGENT_VIEW_SIZE * VIEW_TILE_SIZE)`
                - Dtype is preserved: boolean masks remain boolean (used for indexing)
                    and numeric masks keep their dtype (useful for arithmetic operations).
        """
        original_shape = original_matrix.shape
        original_height, original_width = original_shape[:2]
        new_height = original_height * scale_factor
        new_width = original_width * scale_factor

        if len(original_shape) == 2:
            # Create a new 2D matrix with the expanded size
            expanded_matrix = np.zeros((new_height, new_width), dtype=original_matrix.dtype)

            # Fill the new matrix by expanding each pixel
            for i in range(original_height):
                for j in range(original_width):
                    expanded_matrix[i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor] = original_matrix[i, j]

        elif len(original_shape) == 3:
            # Handle 3D matrices, assuming third dimension is channel (e.g., RGB)
            channels = original_shape[2]
            expanded_matrix = np.zeros((new_height, new_width, channels), dtype=original_matrix.dtype)

            # Fill the new matrix by expanding each pixel for each channel
            for i in range(original_height):
                for j in range(original_width):
                    expanded_matrix[i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor, :] = original_matrix[i, j, :]

        return expanded_matrix
    
    # Grid building functions
    def put_obj_rect(self, obj, topX, topY, width, height):
        for x in range(topX, topX + width):
            for y in range(topY, topY + height):
                self.grid.set(x, y, obj)

    def put_obj_horz(self, obj, topX, topY, width):
        for x in range(topX, topX + width):
            self.grid.set(x, topY, obj)

    def put_obj_vert(self, obj, topX, topY, height):
        for y in range(topY, topY + height):
            self.grid.set(topX, y, obj)

    def update_grid_configuration(self, grid_configuration):
        # update the grid with the new configuration
        for i, bl in enumerate(BARRIER_LOCATIONS):
            if grid_configuration[i+1] != self.maze_state_array[i+1]:
                if grid_configuration[i+1] == 1:
                    self.put_obj(Barrier(), bl[0], bl[1])
                    self.maze_state_array[i+1] = 1
                else:
                    self.grid.set(bl[0], bl[1], None)
                    self.maze_state_array[i+1] = 0    
        
        for i, cl in enumerate(CUE_LOCATIONS):
            if grid_configuration[i+17] != self.maze_state_array[i+17]:
                if grid_configuration[i+17] == 1:
                    self.put_obj(Wall(color='cue_on_rgb'), cl[0], cl[1])
                    self.maze_state_array[i+17] = 1
                else:
                    self.put_obj(Wall(color='cue_off_rgb'), cl[0], cl[1])
                    self.maze_state_array[i+17] = 0

        for i, wl in enumerate(WELL_LOCATIONS):
            if grid_configuration[i+21] != self.maze_state_array[i+21]:
                if grid_configuration[i+21] == 1:
                    self.put_obj(RewardWell(), wl[0], wl[1])
                    self.maze_state_array[i+21] = 1
                else:
                    self.put_obj(EmptyWell(), wl[0], wl[1])
                    self.maze_state_array[i+21] = 0

        for i, tl in enumerate(TRIGGER_LOCATIONS):
            
            if grid_configuration[i+25] != self.maze_state_array[i+25]:
                if grid_configuration[i+25] == 1:
                    self.put_obj(Trigger(trigger_type='A'), tl[0], tl[1])
                    self.maze_state_array[i+25] = 1
                elif grid_configuration[i+25] == 2:
                    self.put_obj(Trigger(trigger_type='B'), tl[0], tl[1])
                    self.maze_state_array[i+25] = 2
                elif grid_configuration[i+25] == 3:
                    self.put_obj(Trigger(trigger_type='S'), tl[0], tl[1])
                    self.maze_state_array[i+25] = 3
                else:
                    self.grid.set(tl[0], tl[1], None)
                    self.maze_state_array[i+25] = 0
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # Reset action control vairaibles
        self.last_action = None
        self.key_actions = 0

        # reset turn score and trial score
        self.turn_score = [None, None]
        self.trial_score = None
        self.session_phase = None
        self.episode_trial_scores = []
        self.episode_turn_scores = []
        self.session_num_trials = None
        self.trial_count = 0 # tracks current trial
        self.sequence_count = 0 # used to track sequence position
        self.phase_punishment_scr = 0
        self.phase_step_count = 0
        self.session_reward = 0

        # Clear session path list
        self.trajectory = []

        # Clear position varibles
        self.fwd_pos = None
        self.fwd_cell = None
        self.cur_cell = None

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Track terminated and truncated states without figuring out how to access these from inherited classes
        # TODO: figure out how to access these from inherited classes
        self.episode_terminated = None
        self.episode_truncated = None

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        #obs = self.gen_obs()
        img = self.get_pov_render_mod(tile_size=VIEW_TILE_SIZE)
        
        #self.plot_observation(img)
        obs_mod = {'image': img, 'direction': self.agent_dir, 'mission': self.mission}
        info = {'action_mask': self.get_action_mask()}
        return obs_mod, info

    # Grid configuration sequence generating function: 
    # generates the sequence of grid configurations for the session
    def gen_grid_configuration_sequence(self):
        # Set goal location index to generate episode sequence
        if self.start_goal_location == 'NE':
            goal_location_index = 0
        elif self.start_goal_location == 'SE':
            goal_location_index = 1
        elif self.start_goal_location == 'SW':
            goal_location_index = 2
        elif self.start_goal_location == 'NW':
            goal_location_index = 3
        elif self.start_goal_location == 'random':
            goal_location_index = random.randint(0,3)
        else:
            goal_location_index = random.randint(0,3)
       
        # returns list start goal pairs (start arm, goal)
        def gen_pi_vc_f2_single_trial():
            # Cue is always in the north postion for this session type and goal is adjusted based
            # cue goal orientation
            # sgc element: (start arm, cue, goal location index) 
            if goal_location_index == 0:
                sgc_list = [(random.choice([1, 3]), 0, 0)]
            elif goal_location_index == 1:
                sgc_list = [(random.choice([1, 3]), 0, 1)]
            elif goal_location_index == 2:
                sgc_list = [(random.choice([1, 3]), 0, 2)]
            elif goal_location_index == 3:
                sgc_list = [(random.choice([1, 3]), 0, 3)]

            return sgc_list

        def gen_pi_vc_f2_acq():
            # Acquisition session are a total of 32 trials and shuffled in chunks of 16 trials
            chunk_size = 4
            # direct route to goal is added for easy checking of repeated routes and then removed  when done creating list
            # LL:0, RR:1, RL:2, LR:3
            # (start arm, cue, route, goal location index)
            # print 
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((goal_location_index + 1) % 4, goal_location_index, 1, goal_location_index), 
                            ((goal_location_index - 1) % 4, goal_location_index, 3, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [(goal_location_index, (goal_location_index - 1) % 4, 0, goal_location_index), 
                            ((goal_location_index + 2) % 4, (goal_location_index - 1) % 4, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [((goal_location_index - 1) % 4, (goal_location_index - 2) % 4, 3, goal_location_index), 
                            ((goal_location_index + 1) % 4, (goal_location_index - 2) % 4, 1, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [((goal_location_index + 2) % 4, (goal_location_index + 1) % 4, 2, goal_location_index), 
                            (goal_location_index, (goal_location_index + 1) % 4, 0, goal_location_index)]
            sg_pairs_temp = sgc_list * 4

            start_repeat_limit = 2
            start_threepeat = 0
            start_fourpeat = 0
            start_repeat_loc = []
            sgc_list = []
            passed = False
            fails = 0
            while not passed:
                passed = True
                sgc_list.clear()
                for _ in range(chunk_size):
                    random.shuffle(sg_pairs_temp)
                    sgc_list += sg_pairs_temp.copy()
                
                for i, sgp in enumerate(sgc_list[0:-3]):
                    if sgp[0] == sgc_list[i + 1][0] and sgp[0] == sgc_list[i + 2][0] and sgp[0] == sgc_list[i + 3][0]:
                        start_fourpeat += 1
                    if sgp[0] == sgc_list[i + 1][0] and sgp[0] == sgc_list[i + 2][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                    if i == len(sgc_list) - 4 and sgc_list[i + 1][0] == sgc_list[i + 2][0] and sgc_list[i + 1][0] == \
                            sgc_list[i + 3][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                # Catch four repeats or over threepeat limit
                if (start_fourpeat > 0 or start_threepeat > start_repeat_limit):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue

                # Catch route threepeats that are too close together for first occurance
                if (len(start_repeat_loc) == 3 and ((start_repeat_loc[1] - start_repeat_loc[0]) < 16) or
                        (len(start_repeat_loc) == 2 and ((start_repeat_loc[1] - start_repeat_loc[0]) < 16))):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(start_repeat_loc) == 3 and (start_repeat_loc[2] - start_repeat_loc[1] < 16):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
            
            #print(f'Length List: {len(sgc_list)}')
            return [(m,n,p) for m, n, o, p in sgc_list]

        def gen_pi_vc_f2_novel_route():
            # The novel route probe session is 40 trials with the first 16 trials being acquisition
            # trials followed by a mix of 16 novel route trials interleaved with 8 acquisition
            # trials (4 and 4 from each arm). index_size is the total trials and chunks is the
            # number mini list to shuffle
            chunk_size = 6
            # direct route to goal is added for easy checking of repeated routes and then removed  when done creating list
            # LL:0, RR:1, RL:2, LR:3
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((goal_location_index + 1) % 4, goal_location_index, 1, goal_location_index), 
                            ((goal_location_index - 1) % 4, goal_location_index, 3, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [(goal_location_index, (goal_location_index - 1) % 4, 0, goal_location_index), 
                            ((goal_location_index + 2) % 4, (goal_location_index - 1) % 4, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [((goal_location_index - 1) % 4, (goal_location_index - 2) % 4, 3, goal_location_index), 
                            ((goal_location_index + 1) % 4, (goal_location_index - 2) % 4, 1, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [((goal_location_index + 2) % 4, (goal_location_index + 1) % 4, 2, goal_location_index), 
                            (goal_location_index, (goal_location_index + 1) % 4, 0, goal_location_index)]
            sgc_trained = sgc_list * 4
            
            sg_pairs_list = []
            # Adding novel route from start arm facing cue
            # LL:0, RR:1, RL:2, LR:3 
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sg_pairs_list = [((goal_location_index + 1) % 4, goal_location_index, 1, goal_location_index), 
                                 ((goal_location_index - 1) % 4, goal_location_index, 3, goal_location_index),
                                 ((goal_location_index + 2) % 4, goal_location_index, 2, goal_location_index), 
                                 ((goal_location_index + 2) % 4, goal_location_index, 2, goal_location_index),
                                 ((goal_location_index + 2) % 4, goal_location_index, 2, goal_location_index), 
                                 ((goal_location_index + 2) % 4, goal_location_index, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sg_pairs_list = [(goal_location_index, (goal_location_index - 1) % 4, 0, goal_location_index), 
                                 ((goal_location_index + 2) % 4, (goal_location_index - 1) % 4, 2, goal_location_index),
                                 ((goal_location_index + 1) % 4, (goal_location_index - 1) % 4, 1, goal_location_index), 
                                 ((goal_location_index + 1) % 4, (goal_location_index - 1) % 4, 1, goal_location_index),
                                 ((goal_location_index + 1) % 4, (goal_location_index - 1) % 4, 1, goal_location_index), 
                                 ((goal_location_index + 1) % 4, (goal_location_index - 1) % 4, 1, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sg_pairs_list = [((goal_location_index - 1) % 4, (goal_location_index - 2) % 4, 3, goal_location_index), 
                                 ((goal_location_index + 1) % 4, (goal_location_index - 2) % 4, 1, goal_location_index),
                                 (goal_location_index, (goal_location_index - 2) % 4, 0, goal_location_index), 
                                 (goal_location_index, (goal_location_index - 2) % 4, 0, goal_location_index),
                                 (goal_location_index, (goal_location_index - 2) % 4, 0, goal_location_index), 
                                 (goal_location_index, (goal_location_index - 2) % 4, 0, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sg_pairs_list = [((goal_location_index + 2) % 4, (goal_location_index + 1) % 4, 2, goal_location_index), 
                                 (goal_location_index, (goal_location_index + 1) % 4, 0, goal_location_index),
                                 ((goal_location_index - 1) % 4, (goal_location_index + 1) % 4, 3, goal_location_index), 
                                 ((goal_location_index - 1) % 4, (goal_location_index + 1) % 4, 3, goal_location_index),
                                 ((goal_location_index - 1) % 4, (goal_location_index + 1) % 4, 3, goal_location_index), 
                                 ((goal_location_index - 1) % 4, (goal_location_index + 1) % 4, 3, goal_location_index)]
            sg_pairs_test = sg_pairs_list

            start_threepeat_limit = 0
            start_threepeat = 0
            start_fourpeat = 0
            start_repeat_loc = []
            sg_pairs = []
            temp_item = sg_pairs_test[-1]
            passed = False
            fails = 0
            while not passed:
                passed = True
                sg_pairs.clear()
                for i in range(chunk_size):
                    if i <= 1:
                        random.shuffle(sgc_trained)
                        sg_pairs += sgc_trained.copy()
                    elif i == 2:
                        # Make sure after end of acquisition trial the next trial is a probe
                        sg_pairs_test.remove(temp_item)
                        random.shuffle(sg_pairs_test)
                        sg_pairs += [temp_item] + sg_pairs_test.copy()
                        sg_pairs_test.append(temp_item)
                    else:
                        random.shuffle(sg_pairs_test)
                        if sg_pairs[-6:] == sg_pairs_test:
                            passed = False
                            continue
                        else:
                            sg_pairs += sg_pairs_test.copy()
                len_sg_pairs = len(sg_pairs)
                for i, sgp in enumerate(sg_pairs[0:-3]):
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0] and sgp[0] == sg_pairs[i + 3][0]:
                        start_fourpeat += 1
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                    if i == len_sg_pairs - 4 and sg_pairs[i + 1][0] == sg_pairs[i + 2][0] and sg_pairs[i + 1][0] == sg_pairs[i + 3][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                # Catch four repeats or over threepeat limit
                if (start_fourpeat > 0 or start_threepeat > start_threepeat_limit):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
            
            #print(f'Length Total: {len(sg_pairs)}\nLength Train: {len(sgc_trained)}\nLength Probe: {len(sg_pairs_test)}\nChunk Size: {chunk_size}')
            return [(m,n,p) for m, n, o, p in sg_pairs]
        
        def gen_pi_vc_f2_no_cue():
            # No cue probe session PI+VC acquisition subjects 
            chunk_size = 4
            # direct route to goal is added for easy checking of repeated routes and then removed  when done creating list
            # LL:0, RR:1, RL:2, LR:3
            # (start arm, cue, route, goal location index) cue = 4 <- No Cue
            # print 
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((goal_location_index + 1) % 4, 4, 1, goal_location_index), 
                            ((goal_location_index - 1) % 4, 4, 3, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [(goal_location_index, 4, 0, goal_location_index), 
                            ((goal_location_index + 2) % 4, 4, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [((goal_location_index - 1) % 4, 4, 3, goal_location_index), 
                            ((goal_location_index + 1) % 4, 4, 1, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [((goal_location_index + 2) % 4, 4, 2, goal_location_index), 
                            (goal_location_index, 4, 0, goal_location_index)]
            sg_pairs_temp = sgc_list * 4

            start_repeat_limit = 2
            start_threepeat = 0
            start_fourpeat = 0
            start_repeat_loc = []
            sgc_list = []
            passed = False
            fails = 0
            while not passed:
                passed = True
                sgc_list.clear()
                for _ in range(chunk_size):
                    random.shuffle(sg_pairs_temp)
                    sgc_list += sg_pairs_temp.copy()
                
                len_sgc_list = len(sgc_list)
                for i, sgp in enumerate(sgc_list[0:-3]):
                    if sgp[0] == sgc_list[i + 1][0] and sgp[0] == sgc_list[i + 2][0] and sgp[0] == sgc_list[i + 3][0]:
                        start_fourpeat += 1
                    if sgp[0] == sgc_list[i + 1][0] and sgp[0] == sgc_list[i + 2][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                    if i == len_sgc_list - 4 and sgc_list[i + 1][0] == sgc_list[i + 2][0] and sgc_list[i + 1][0] == \
                            sgc_list[i + 3][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                # Catch four repeats or over threepeat limit
                if (start_fourpeat > 0 or start_threepeat > start_repeat_limit):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue

                # Catch route threepeats that are too close together for first occurance
                if (len(start_repeat_loc) == 3 and ((start_repeat_loc[1] - start_repeat_loc[0]) < 16) or
                        (len(start_repeat_loc) == 2 and ((start_repeat_loc[1] - start_repeat_loc[0]) < 16))):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(start_repeat_loc) == 3 and (start_repeat_loc[2] - start_repeat_loc[1] < 16):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
            
            #print(f'Length Total: {len(sgc_list)}\nChunk Size: {chunk_size}')
            return [(m,n,p) for m, n, o, p in sgc_list]

        def gen_pi_vc_f2_rotate():
            # PI+VC session are a total of 16 trials and shuffled as one chunk
            # This is to limit habit development in the rat.
            chunk_size = 2
            # direct route to goal is added for easy checking of repeated routes and then removed  when done creating list
            # LL:0, RR:1, RL:2, LR:3
            # (start arm, cue, route, goal location index)
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((i+1)%4,i,1,i) for i in range(4)] + [((i-1)%4,i,3,i) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [(i, (i-1)%4, 0,i) for i in range(4)] + [((i+2)%4,(i-1)%4, 2,i) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [((i+1)%4,(i+2)%4,1,i) for i in range(4)] + [((i-1)%4,(i+2)%4,3,i) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [(i, (i+1)%4, 0,i) for i in range(4)] + [((i+2)%4,(i+1)%4, 2,i) for i in range(4)]

            sg_pairs_temp = sgc_list

            start_repeat_limit = 3
            goal_repeat_limit = 3
            route_repeat_limit = 3
            start_threepeat = 0
            start_fourpeat = 0
            goal_fourpeat = 0
            goal_threepeat = 0
            route_fourpeat = 0
            route_threepeat = 0
            start_repeat_loc = []
            route_repeat_loc = []
            goal_repeat_loc = []
            sg_pairs = []
            passed = False
            fails = 0
            while not passed:
                passed = True
                sg_pairs.clear()
                for _ in range(chunk_size):
                    random.shuffle(sg_pairs_temp)
                    sg_pairs += sg_pairs_temp.copy()
                
                # catch fourpeats and locations of threepeats - this is to make sure threepeats aren't
                # too close together.
                for i, sgp in enumerate(sg_pairs[0:-3]):
                    if sgp[2] == sg_pairs[i + 1][2] and sgp[2] == sg_pairs[i + 2][2] and sgp[2] == sg_pairs[i + 3][2]:
                        route_fourpeat += 1
                    if sgp[1] == sg_pairs[i + 1][1] and sgp[1] == sg_pairs[i + 2][1] and sgp[1] == sg_pairs[i + 3][1]:
                        goal_fourpeat += 1
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0] and sgp[0] == sg_pairs[i + 3][0]:
                        start_fourpeat += 1
                    if sgp[2] == sg_pairs[i + 1][2] and sgp[2] == sg_pairs[i + 2][2]:
                        route_threepeat += 1
                        route_repeat_loc.append(i)
                    if sgp[1] == sg_pairs[i + 1][1] and sgp[1] == sg_pairs[i + 2][1]:
                        goal_threepeat += 1
                        goal_repeat_loc.append(i)
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                
                # Catch four repeats or over threepeat limit
                if (route_fourpeat > 0 or goal_fourpeat > 0 or start_fourpeat > 0 or
                        route_threepeat > route_repeat_limit or
                        goal_threepeat > goal_repeat_limit or
                        start_threepeat > start_repeat_limit):
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue

                # Catch route threepeats that are too close together for first occurrence
                if (len(route_repeat_loc) == 3 and (route_repeat_loc[1] - route_repeat_loc[0] < 16) or
                        (len(route_repeat_loc) == 2 and route_repeat_loc[1] - route_repeat_loc[0] < 16)):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(route_repeat_loc) == 3 and (route_repeat_loc[2] - route_repeat_loc[1] < 16):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue

                # Catch goal threepeats that are too close together for first occurance
                if (len(goal_repeat_loc) == 3 and (goal_repeat_loc[1] - goal_repeat_loc[0] < 16) or
                        (len(goal_repeat_loc) == 2 and goal_repeat_loc[1] - goal_repeat_loc[0] < 16)):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(goal_repeat_loc) == 3 and (goal_repeat_loc[2] - goal_repeat_loc[1] < 16):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                
                # Catch route threepeats that are too close together for first occurance
                if (len(start_repeat_loc) == 3 and (start_repeat_loc[1] - start_repeat_loc[0] < 16) or
                        (len(start_repeat_loc) == 2 and start_repeat_loc[1] - start_repeat_loc[0] < 16)):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(start_repeat_loc) == 3 and (start_repeat_loc[2] - start_repeat_loc[1] < 16):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
            
            #print(f"list size: {len(sg_pairs)}")   
            return [(m,n,p) for m, n, o, p in sg_pairs]
        
        def gen_pi_vc_f2_reversal():
            # The reversal probe has a total of 80 trials with the first 16 trials being acquisition
            # trials and the subsequent 64 trials being reversals. Each post acquisition trial chunk
            # is size 8.
            chunk_size = 10

            # direct route to goal is added for easy checking of repeated routes and then removed  
            # when done creating list
            # LL:0, RR:1, RL:2, LR:3
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((goal_location_index + 1) % 4, goal_location_index, 1, goal_location_index), 
                            ((goal_location_index - 1) % 4, goal_location_index, 3, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [(goal_location_index, (goal_location_index - 1) % 4, 0, goal_location_index), 
                            ((goal_location_index + 2) % 4, (goal_location_index - 1) % 4, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [((goal_location_index - 1) % 4, (goal_location_index - 2) % 4, 3, goal_location_index), 
                            ((goal_location_index + 1) % 4, (goal_location_index - 2) % 4, 1, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [((goal_location_index + 2) % 4, (goal_location_index + 1) % 4, 2, goal_location_index), 
                            (goal_location_index, (goal_location_index + 1) % 4, 0, goal_location_index)]
            sgc_trained = sgc_list * 4

            sg_pairs_list = []
            # Adding novel route from start arm facing cue
            # LL:0, RR:1, RL:2, LR:3 
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sg_pairs_list = [((goal_location_index + 1) % 4, goal_location_index, 1, (goal_location_index + 2) % 4), 
                                 ((goal_location_index - 1) % 4, goal_location_index, 3, (goal_location_index + 2) % 4)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sg_pairs_list = [(goal_location_index, (goal_location_index - 1) % 4, 0, (goal_location_index + 2) % 4), 
                                 ((goal_location_index + 2) % 4, (goal_location_index - 1) % 4, 2, (goal_location_index + 2) % 4)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sg_pairs_list = [((goal_location_index - 1) % 4, (goal_location_index - 2) % 4, 3, (goal_location_index + 2) % 4), 
                                 ((goal_location_index + 1) % 4, (goal_location_index - 2) % 4, 1, (goal_location_index + 2) % 4)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sg_pairs_list = [((goal_location_index + 2) % 4, (goal_location_index + 1) % 4, 2, (goal_location_index + 2) % 4), 
                                 (goal_location_index, (goal_location_index + 1) % 4, 0, (goal_location_index + 2) % 4)]
            sg_pairs_test = sg_pairs_list * 4

            start_threepeat_limit = 0
            start_threepeat = 0
            start_fourpeat = 0
            start_repeat_loc = []
            sg_pairs = []
            passed = False
            fails = 0
            while not passed:
                passed = True
                sg_pairs.clear()
                # Shuffle lists in separate batches, of acquisition and probe portions
                for i in range(chunk_size):
                    if i <= 1:
                        random.shuffle(sgc_trained)
                        sg_pairs += sgc_trained.copy()
                    else:
                        random.shuffle(sg_pairs_test)
                        sg_pairs += sg_pairs_test.copy()
                len_sg_pairs = len(sg_pairs)
                for i, sgp in enumerate(sg_pairs[0:-3]):
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0] and sgp[0] == sg_pairs[i + 3][0]:
                        start_fourpeat += 1
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                    if i == len_sg_pairs - 4 and sg_pairs[i + 1][0] == sg_pairs[i + 2][0] and sg_pairs[i + 1][0] == sg_pairs[i + 3][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                # Catch four repeats or over threepeat limit
                if (start_fourpeat > 0 or start_threepeat > start_threepeat_limit):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
            
            #print(f'Length Total: {len(sg_pairs)}\nLength Train: {len(sgc_trained)}\nLength Probe: {len(sg_pairs_test)}\nChunk Size: {chunk_size}')
            return [(m,n,p) for m, n, o, p in sg_pairs]

        def gen_pi_vc_f1_acq():
            # Acquisition session are a total of 32 trials and shuffled in chunks of 16 trials
            chunk_size = 4
            # direct route to goal is added for easy checking of repeated routes and then removed  when done creating list
            # LL:0, RR:1, RL:2, LR:3
            # (start arm, cue, route, goal location index)
            # print 
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((goal_location_index + 1) % 4, goal_location_index, 1, goal_location_index), 
                            ((goal_location_index + 2) % 4, goal_location_index, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [((goal_location_index+1)%4, (goal_location_index-1)%4, 1, goal_location_index), 
                            ((goal_location_index+2)%4, (goal_location_index-1)%4, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [((goal_location_index-1)%4, (goal_location_index+2)%4, 3, goal_location_index), 
                            (goal_location_index, (goal_location_index+2)%4, 0, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [((goal_location_index-1)%4, (goal_location_index+1)%4, 3, goal_location_index), 
                            (goal_location_index, (goal_location_index+1)%4, 0, goal_location_index)]
            sg_pairs_temp = sgc_list * 4

            start_repeat_limit = 2
            start_threepeat = 0
            start_fourpeat = 0
            start_repeat_loc = []
            sgc_list = []
            passed = False
            fails = 0
            while not passed:
                passed = True
                sgc_list.clear()
                for _ in range(chunk_size):
                    random.shuffle(sg_pairs_temp)
                    sgc_list += sg_pairs_temp.copy()
                len_sgc_list = len(sgc_list)
                for i, sgp in enumerate(sgc_list[0:-3]):
                    if sgp[0] == sgc_list[i + 1][0] and sgp[0] == sgc_list[i + 2][0] and sgp[0] == sgc_list[i + 3][0]:
                        start_fourpeat += 1
                    if sgp[0] == sgc_list[i + 1][0] and sgp[0] == sgc_list[i + 2][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                    if i == len_sgc_list - 4 and sgc_list[i + 1][0] == sgc_list[i + 2][0] and sgc_list[i + 1][0] == \
                            sgc_list[i + 3][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                # Catch four repeats or over threepeat limit
                if (start_fourpeat > 0 or start_threepeat > start_repeat_limit):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue

                # Catch route threepeats that are too close together for first occurance
                if (len(start_repeat_loc) == 3 and ((start_repeat_loc[1] - start_repeat_loc[0]) < 16) or
                        (len(start_repeat_loc) == 2 and ((start_repeat_loc[1] - start_repeat_loc[0]) < 16))):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(start_repeat_loc) == 3 and (start_repeat_loc[2] - start_repeat_loc[1] < 16):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
            
            #print(f'Length List: {len(sgc_list)}')
            return [(m,n,p) for m, n, o, p in sgc_list]

        def gen_pi_vc_f1_novel_route():
            # trained on f1 (fixed first turn with alternating second turn) novel route with have
            # an alternate first turn.
            # The novel route probe session is 40 trials with the first 16 trials being acquisition
            # trials followed by a mix of 16 novel route trials interleaved with 8 acquisition
            # trials (4 and 4 from each arm). index_size is the total trials and chunks is the
            # number mini list to shuffle
            chunk_size = 6
            # direct route to goal is added for easy checking of repeated routes and then removed  when done creating list
            # LL:0, RR:1, RL:2, LR:3
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((goal_location_index + 1) % 4, goal_location_index, 1, goal_location_index), 
                            ((goal_location_index + 2) % 4, goal_location_index, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [((goal_location_index+1)%4, (goal_location_index-1)%4, 1, goal_location_index), 
                            ((goal_location_index+2)%4, (goal_location_index-1)%4, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [((goal_location_index-1)%4, (goal_location_index+2)%4, 3, goal_location_index), 
                            (goal_location_index, (goal_location_index+2)%4, 0, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [((goal_location_index-1)%4, (goal_location_index+1)%4, 3, goal_location_index), 
                            (goal_location_index, (goal_location_index+1)%4, 0, goal_location_index)]
            sgc_trained = sgc_list * 4

            sg_pairs_list = []
            # Adding novel route from start arm facing cue
            # LL:0, RR:1, RL:2, LR:3 
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sg_pairs_list = [((goal_location_index + 1) % 4, goal_location_index, 1, goal_location_index), 
                                 ((goal_location_index + 2) % 4, goal_location_index, 2, goal_location_index),
                                 ((goal_location_index - 1) % 4, goal_location_index, 3, goal_location_index), 
                                 ((goal_location_index - 1) % 4, goal_location_index, 3, goal_location_index),
                                 ((goal_location_index - 1) % 4, goal_location_index, 3, goal_location_index), 
                                 ((goal_location_index - 1) % 4, goal_location_index, 3, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sg_pairs_list = [((goal_location_index + 1) % 4, (goal_location_index - 1) % 4, 1, goal_location_index), 
                                 ((goal_location_index + 2) % 4, (goal_location_index - 1) % 4, 2, goal_location_index),
                                 (goal_location_index, (goal_location_index - 1) % 4, 0, goal_location_index), 
                                 (goal_location_index, (goal_location_index - 1) % 4, 0, goal_location_index),
                                 (goal_location_index, (goal_location_index - 1) % 4, 0, goal_location_index), 
                                 (goal_location_index, (goal_location_index - 1) % 4, 0, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sg_pairs_list = [((goal_location_index - 1) % 4, (goal_location_index - 2) % 4, 3, goal_location_index), 
                                 (goal_location_index, (goal_location_index - 2) % 4, 0, goal_location_index),
                                 ((goal_location_index + 1) % 4, (goal_location_index - 2) % 4, 1, goal_location_index), 
                                 ((goal_location_index + 1) % 4, (goal_location_index - 2) % 4, 1, goal_location_index),
                                 ((goal_location_index + 1) % 4, (goal_location_index - 2) % 4, 1, goal_location_index), 
                                 ((goal_location_index + 1) % 4, (goal_location_index - 2) % 4, 1, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sg_pairs_list = [((goal_location_index - 1) % 4, (goal_location_index + 1) % 4, 3, goal_location_index), 
                                 (goal_location_index, (goal_location_index + 1) % 4, 0, goal_location_index),
                                 ((goal_location_index + 2) % 4, (goal_location_index + 1) % 4, 2, goal_location_index), 
                                 ((goal_location_index + 2) % 4, (goal_location_index + 1) % 4, 2, goal_location_index),
                                 ((goal_location_index + 2) % 4, (goal_location_index + 1) % 4, 2, goal_location_index), 
                                 ((goal_location_index + 2) % 4, (goal_location_index + 1) % 4, 2, goal_location_index)]
            sg_pairs_test = sg_pairs_list

            start_threepeat_limit = 0
            start_threepeat = 0
            start_fourpeat = 0
            start_repeat_loc = []
            sg_pairs = []
            temp_item = sg_pairs_test[-1]
            passed = False
            fails = 0
            while not passed:
                passed = True
                sg_pairs.clear()
                for i in range(chunk_size):
                    if i <= 1:
                        random.shuffle(sgc_trained)
                        sg_pairs += sgc_trained.copy()
                    elif i == 2:
                        # Make sure after end of acquisition trial the next trial is a probe
                        sg_pairs_test.remove(temp_item)
                        random.shuffle(sg_pairs_test)
                        sg_pairs += [temp_item] + sg_pairs_test.copy()
                        sg_pairs_test.append(temp_item)
                    else:
                        random.shuffle(sg_pairs_test)
                        if sg_pairs[-6:] == sg_pairs_test:
                            passed = False
                            continue
                        else:
                            sg_pairs += sg_pairs_test.copy()
                
                len_sg_pairs = len(sg_pairs)
                for i, sgp in enumerate(sg_pairs[0:-3]):
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0] and sgp[0] == sg_pairs[i + 3][0]:
                        start_fourpeat += 1
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                    if i == len_sg_pairs - 4 and sg_pairs[i + 1][0] == sg_pairs[i + 2][0] and sg_pairs[i + 1][0] == sg_pairs[i + 3][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                # Catch four repeats or over threepeat limit
                if (start_fourpeat > 0 or start_threepeat > start_threepeat_limit):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
            
            #print(f'Length Total: {len(sg_pairs)}\nLength Train: {len(sgc_trained)}\nLength Probe: {len(sg_pairs_test)}\nChunk Size: {chunk_size}')
            return [(m,n,p) for m, n, o, p in sg_pairs]
      
        def gen_pi_vc_f1_no_cue():
            chunk_size = 4
            # direct route to goal is added for easy checking of repeated routes and then removed  when done creating list
            # LL:0, RR:1, RL:2, LR:3
            # (start arm, cue, route, goal location index) cue = 4 <- No Cue
            # print 
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((goal_location_index + 1) % 4, 4, 1, goal_location_index), 
                            ((goal_location_index + 2) % 4, 4, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [((goal_location_index+1)%4, 4, 1, goal_location_index), 
                            ((goal_location_index+2)%4, 4, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [((goal_location_index-1)%4, 4, 3, goal_location_index), 
                            (goal_location_index, 4, 0, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [((goal_location_index-1)%4, 4, 3, goal_location_index), 
                            (goal_location_index, 4, 0, goal_location_index)]
            sg_pairs_temp = sgc_list * 4

            start_repeat_limit = 2
            start_threepeat = 0
            start_fourpeat = 0
            start_repeat_loc = []
            sgc_list = []
            passed = False
            fails = 0
            while not passed:
                passed = True
                sgc_list.clear()
                for _ in range(chunk_size):
                    random.shuffle(sg_pairs_temp)
                    sgc_list += sg_pairs_temp.copy()
                
                len_sgc_list = len(sgc_list)
                for i, sgp in enumerate(sgc_list[0:-3]):
                    if sgp[0] == sgc_list[i + 1][0] and sgp[0] == sgc_list[i + 2][0] and sgp[0] == sgc_list[i + 3][0]:
                        start_fourpeat += 1
                    if sgp[0] == sgc_list[i + 1][0] and sgp[0] == sgc_list[i + 2][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                    if i == len_sgc_list - 4 and sgc_list[i + 1][0] == sgc_list[i + 2][0] and sgc_list[i + 1][0] == \
                            sgc_list[i + 3][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                # Catch four repeats or over threepeat limit
                if (start_fourpeat > 0 or start_threepeat > start_repeat_limit):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue

                # Catch route threepeats that are too close together for first occurance
                if (len(start_repeat_loc) == 3 and ((start_repeat_loc[1] - start_repeat_loc[0]) < 16) or
                        (len(start_repeat_loc) == 2 and ((start_repeat_loc[1] - start_repeat_loc[0]) < 16))):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(start_repeat_loc) == 3 and (start_repeat_loc[2] - start_repeat_loc[1] < 16):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
            
            #print(f'Length Total: {len(sgc_list)}\nChunk Size: {chunk_size}')
            return [(m,n,p) for m, n, o, p in sgc_list]

        def gen_pi_vc_f1_rotate():
            # PI+VC session are a total of 16 trials and shuffled as one chunk
            # This is to limit habit development in the rat.
            chunk_size = 2
            # direct route to goal is added for easy checking of repeated routes and then removed  when done creating list
            # LL:0, RR:1, RL:2, LR:3
            # (start arm, cue, route, goal location index)
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((i+1)%4,i,1,i) for i in range(4)] + [((i+2)%4,i,2,i) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [((i+1)%4, (i-1)%4, 1,i) for i in range(4)] + [((i+2)%4,(i-1)%4, 2,i) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [(i,(i+2)%4,0,i) for i in range(4)] + [((i-1)%4,(i+2)%4,3,i) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [(i, (i+1)%4, 0,i) for i in range(4)] + [((i-1)%4,(i+1)%4, 3,i) for i in range(4)]
            sg_pairs_temp = sgc_list

            start_repeat_limit = 3
            goal_repeat_limit = 3
            route_repeat_limit = 3
            start_threepeat = 0
            start_fourpeat = 0
            goal_fourpeat = 0
            goal_threepeat = 0
            route_fourpeat = 0
            route_threepeat = 0
            start_repeat_loc = []
            route_repeat_loc = []
            goal_repeat_loc = []
            sg_pairs = []
            passed = False
            fails = 0
            while not passed:
                passed = True
                sg_pairs.clear()
                for _ in range(chunk_size):
                    random.shuffle(sg_pairs_temp)
                    sg_pairs += sg_pairs_temp.copy()
                
                # catch fourpeats and locations of threepeats - this is to make sure threepeats aren't
                # too close together.
                for i, sgp in enumerate(sg_pairs[0:-3]):
                    if sgp[2] == sg_pairs[i + 1][2] and sgp[2] == sg_pairs[i + 2][2] and sgp[2] == sg_pairs[i + 3][2]:
                        route_fourpeat += 1
                    if sgp[1] == sg_pairs[i + 1][1] and sgp[1] == sg_pairs[i + 2][1] and sgp[1] == sg_pairs[i + 3][1]:
                        goal_fourpeat += 1
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0] and sgp[0] == sg_pairs[i + 3][0]:
                        start_fourpeat += 1
                    if sgp[2] == sg_pairs[i + 1][2] and sgp[2] == sg_pairs[i + 2][2]:
                        route_threepeat += 1
                        route_repeat_loc.append(i)
                    if sgp[1] == sg_pairs[i + 1][1] and sgp[1] == sg_pairs[i + 2][1]:
                        goal_threepeat += 1
                        goal_repeat_loc.append(i)
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                
                # Catch four repeats or over threepeat limit
                if (route_fourpeat > 0 or goal_fourpeat > 0 or start_fourpeat > 0 or
                        route_threepeat > route_repeat_limit or
                        goal_threepeat > goal_repeat_limit or
                        start_threepeat > start_repeat_limit):
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue

                # Catch route threepeats that are too close together for first occurrence
                if (len(route_repeat_loc) == 3 and (route_repeat_loc[1] - route_repeat_loc[0] < 16) or
                        (len(route_repeat_loc) == 2 and route_repeat_loc[1] - route_repeat_loc[0] < 16)):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(route_repeat_loc) == 3 and (route_repeat_loc[2] - route_repeat_loc[1] < 16):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue

                # Catch goal threepeats that are too close together for first occurance
                if (len(goal_repeat_loc) == 3 and (goal_repeat_loc[1] - goal_repeat_loc[0] < 16) or
                        (len(goal_repeat_loc) == 2 and goal_repeat_loc[1] - goal_repeat_loc[0] < 16)):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(goal_repeat_loc) == 3 and (goal_repeat_loc[2] - goal_repeat_loc[1] < 16):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                
                # Catch route threepeats that are too close together for first occurance
                if (len(start_repeat_loc) == 3 and (start_repeat_loc[1] - start_repeat_loc[0] < 16) or
                        (len(start_repeat_loc) == 2 and start_repeat_loc[1] - start_repeat_loc[0] < 16)):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(start_repeat_loc) == 3 and (start_repeat_loc[2] - start_repeat_loc[1] < 16):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
            
            #print(f'Length Total: {len(sg_pairs)}\nChunk Size: {chunk_size}')       
            return [(m,n,p) for m, n, o, p in sg_pairs]

        def gen_pi_vc_f1_reversal():
            # The reversal probe has a total of 80 trials with the first 16 trials being acquisition
            # trials and the subsequent 64 trials being reversals. Each post acquisition trial chunk
            # is size 8.
            chunk_size = 10

            # direct route to goal is added for easy checking of repeated routes and then removed  
            # when done creating list
            # LL:0, RR:1, RL:2, LR:3
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((goal_location_index + 1) % 4, goal_location_index, 1, goal_location_index), 
                            ((goal_location_index + 2) % 4, goal_location_index, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [((goal_location_index+1)%4, (goal_location_index-1)%4, 1, goal_location_index), 
                            ((goal_location_index+2)%4, (goal_location_index-1)%4, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [((goal_location_index-1)%4, (goal_location_index+2)%4, 3, goal_location_index), 
                            (goal_location_index, (goal_location_index+2)%4, 0, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [((goal_location_index-1)%4, (goal_location_index+1)%4, 3, goal_location_index), 
                            (goal_location_index, (goal_location_index+1)%4, 0, goal_location_index)]
            sgc_trained = sgc_list * 4

            # Adding novel route from start arm facing cue
            # LL:0, RR:1, RL:2, LR:3 
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((goal_location_index + 1) % 4, goal_location_index, 1, (goal_location_index+2)%4), 
                            ((goal_location_index + 2) % 4, goal_location_index, 2, (goal_location_index+2)%4)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [((goal_location_index+1)%4, (goal_location_index-1)%4, 1, (goal_location_index+2)%4), 
                            ((goal_location_index+2)%4, (goal_location_index-1)%4, 2, (goal_location_index+2)%4)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [((goal_location_index-1)%4, (goal_location_index+2)%4, 3, (goal_location_index+2)%4), 
                            (goal_location_index, (goal_location_index+2)%4, 0, (goal_location_index+2)%4)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [((goal_location_index-1)%4, (goal_location_index+1)%4, 3, (goal_location_index+2)%4), 
                            (goal_location_index, (goal_location_index+1)%4, 0, (goal_location_index+2)%4)]
            sg_pairs_test = sgc_list * 4

            start_threepeat_limit = 0
            start_threepeat = 0
            start_fourpeat = 0
            start_repeat_loc = []
            sg_pairs = []
            passed = False
            fails = 0
            while not passed:
                passed = True
                sg_pairs.clear()
                # Shuffle lists in separate batches, of acquisition and probe portions
                for i in range(chunk_size):
                    if i <= 1:
                        random.shuffle(sgc_trained)
                        sg_pairs += sgc_trained.copy()
                    else:
                        random.shuffle(sg_pairs_test)
                        sg_pairs += sg_pairs_test.copy()
                len_sg_pairs = len(sg_pairs)
                for i, sgp in enumerate(sg_pairs[0:-3]):
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0] and sgp[0] == sg_pairs[i + 3][0]:
                        start_fourpeat += 1
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                    if i == len_sg_pairs - 4 and sg_pairs[i + 1][0] == sg_pairs[i + 2][0] and sg_pairs[i + 1][0] == sg_pairs[i + 3][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                # Catch four repeats or over threepeat limit
                if (start_fourpeat > 0 or start_threepeat > start_threepeat_limit):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
            
            #print(f'Length Total: {len(sg_pairs)}\nLength Train: {len(sgc_trained)}\nLength Probe: {len(sg_pairs_test)}\nChunk Size: {chunk_size}')
            return [(m,n,p) for m, n, o, p in sg_pairs]

        def gen_pi_novel_route_no_cue():
            # The novel route probe session is 40 trials with the first 16 trials being acquisition
            # trials followed by a mix of 16 novel route trials interleaved with 8 acquisition
            # trials (4 and 4 from each arm). index_size is the total trials and chunks is the
            # number mini list to shuffle. This session uses no cue
            chunk_size = 6
            # direct route to goal is added for easy checking of repeated routes and then removed  when done creating list
            # LL:0, RR:1, RL:2, LR:3
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((goal_location_index + 1) % 4, 4, 1, goal_location_index), 
                            ((goal_location_index - 1) % 4, 4, 3, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [(goal_location_index, 4, 0, goal_location_index), 
                            ((goal_location_index + 2) % 4, 4, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [((goal_location_index - 1) % 4, 4, 3, goal_location_index), 
                            ((goal_location_index + 1) % 4, 4, 1, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [((goal_location_index + 2) % 4, 4, 2, goal_location_index), 
                            (goal_location_index, 4, 0, goal_location_index)]
            sgc_trained = sgc_list * 4

            sg_pairs_list = []
            # Adding novel route from start arm facing cue
            # LL:0, RR:1, RL:2, LR:3 
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sg_pairs_list = [((goal_location_index + 1) % 4, 4, 1, goal_location_index), 
                                 ((goal_location_index - 1) % 4, 4, 3, goal_location_index),
                                 ((goal_location_index + 2) % 4, 4, 2, goal_location_index), 
                                 ((goal_location_index + 2) % 4, 4, 2, goal_location_index),
                                 ((goal_location_index + 2) % 4, 4, 2, goal_location_index), 
                                 ((goal_location_index + 2) % 4, 4, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sg_pairs_list = [(goal_location_index, 4, 0, goal_location_index), 
                                 ((goal_location_index + 2) % 4, 4, 2, goal_location_index),
                                 ((goal_location_index + 1) % 4, 4, 1, goal_location_index), 
                                 ((goal_location_index + 1) % 4, 4, 1, goal_location_index),
                                 ((goal_location_index + 1) % 4, 4, 1, goal_location_index), 
                                 ((goal_location_index + 1) % 4, 4, 1, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sg_pairs_list = [((goal_location_index - 1) % 4, 4, 3, goal_location_index), 
                                 ((goal_location_index + 1) % 4, 4, 1, goal_location_index),
                                 (goal_location_index, 4, 0, goal_location_index), 
                                 (goal_location_index, 4, 0, goal_location_index),
                                 (goal_location_index, 4, 0, goal_location_index), 
                                 (goal_location_index, 4, 0, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sg_pairs_list = [((goal_location_index + 2) % 4, 4, 2, goal_location_index), 
                                 (goal_location_index, 4, 0, goal_location_index),
                                 ((goal_location_index - 1) % 4, 4, 3, goal_location_index), 
                                 ((goal_location_index - 1) % 4, 4, 3, goal_location_index),
                                 ((goal_location_index - 1) % 4, 4, 3, goal_location_index), 
                                 ((goal_location_index - 1) % 4, 4, 3, goal_location_index)]
            sg_pairs_test = sg_pairs_list

            start_threepeat_limit = 0
            start_threepeat = 0
            start_fourpeat = 0
            start_repeat_loc = []
            sg_pairs = []
            temp_item = sg_pairs_test[-1]
            passed = False
            fails = 0
            while not passed:
                passed = True
                sg_pairs.clear()
                for i in range(chunk_size):
                    if i <= 1:
                        random.shuffle(sgc_trained)
                        sg_pairs += sgc_trained.copy()
                    elif i == 2:
                        # Make sure after end of acquisition trial the next trial is a probe
                        sg_pairs_test.remove(temp_item)
                        random.shuffle(sg_pairs_test)
                        sg_pairs += [temp_item] + sg_pairs_test.copy()
                        sg_pairs_test.append(temp_item)
                    else:
                        random.shuffle(sg_pairs_test)
                        if sg_pairs[-6:] == sg_pairs_test:
                            passed = False
                            continue
                        else:
                            sg_pairs += sg_pairs_test.copy()
                len_sg_pairs = len(sg_pairs)
                for i, sgp in enumerate(sg_pairs[0:-3]):
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0] and sgp[0] == sg_pairs[i + 3][0]:
                        start_fourpeat += 1
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                    if i == len_sg_pairs - 4 and sg_pairs[i + 1][0] == sg_pairs[i + 2][0] and sg_pairs[i + 1][0] == sg_pairs[i + 3][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                # Catch four repeats or over threepeat limit
                if (start_fourpeat > 0 or start_threepeat > start_threepeat_limit):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
            
            #print(f'Length Total: {len(sg_pairs)}\nLength Train: {len(sgc_trained)}\nLength Probe: {len(sg_pairs_test)}\nChunk Size: {chunk_size}')
            return [(m,n,p) for m, n, o, p in sg_pairs]        

        def gen_pi_reversal_no_cue():
            # The reversal probe has a total of 80 trials with the first 16 trials being acquisition
            # trials and the subsequent 64 trials being reversals. Each post acquisition trial chunk
            # is size 8.
            chunk_size = 10

            # direct route to goal is added for easy checking of repeated routes and then removed  
            # when done creating list
            # LL:0, RR:1, RL:2, LR:3
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((goal_location_index + 1) % 4, 4, 1, goal_location_index), 
                            ((goal_location_index - 1) % 4, 4, 3, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [(goal_location_index, 4, 0, goal_location_index), 
                            ((goal_location_index + 2) % 4, 4, 2, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [((goal_location_index - 1) % 4, 4, 3, goal_location_index), 
                            ((goal_location_index + 1) % 4, 4, 1, goal_location_index)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [((goal_location_index + 2) % 4, 4, 2, goal_location_index), 
                            (goal_location_index, 4, 0, goal_location_index)]
            sgc_trained = sgc_list * 4

            sg_pairs_list = []
            # Adding novel route from start arm facing cue
            # LL:0, RR:1, RL:2, LR:3 
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sg_pairs_list = [((goal_location_index + 1) % 4, 4, 1, (goal_location_index + 2) % 4), 
                                 ((goal_location_index - 1) % 4, 4, 3, (goal_location_index + 2) % 4)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sg_pairs_list = [(goal_location_index, 4, 0, (goal_location_index + 2) % 4), 
                                 ((goal_location_index + 2) % 4, 4, 2, (goal_location_index + 2) % 4)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sg_pairs_list = [((goal_location_index - 1) % 4, 4, 3, (goal_location_index + 2) % 4), 
                                 ((goal_location_index + 1) % 4, 4, 1, (goal_location_index + 2) % 4)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sg_pairs_list = [((goal_location_index + 2) % 4, 4, 2, (goal_location_index + 2) % 4), 
                                 (goal_location_index, 4, 0, (goal_location_index + 2) % 4)]
            sg_pairs_test = sg_pairs_list * 4

            start_threepeat_limit = 0
            start_threepeat = 0
            start_fourpeat = 0
            start_repeat_loc = []
            sg_pairs = []
            passed = False
            fails = 0
            while not passed:
                passed = True
                sg_pairs.clear()
                # Shuffle lists in separate batches, of acquisition and probe portions
                for i in range(chunk_size):
                    if i <=1:
                        random.shuffle(sgc_trained)
                        sg_pairs += sgc_trained.copy()
                    else:
                        random.shuffle(sg_pairs_test)
                        sg_pairs += sg_pairs_test.copy()
                len_sg_pairs = len(sg_pairs)
                for i, sgp in enumerate(sg_pairs[0:-3]):
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0] and sgp[0] == sg_pairs[i + 3][0]:
                        start_fourpeat += 1
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                    if i == len_sg_pairs - 4 and sg_pairs[i + 1][0] == sg_pairs[i + 2][0] and sg_pairs[i + 1][0] == sg_pairs[i + 3][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                # Catch four repeats or over threepeat limit
                if (start_fourpeat > 0 or start_threepeat > start_threepeat_limit):
                    start_threepeat = 0
                    start_fourpeat = 0
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
            
            print(f'Length Total: {len(sg_pairs)}\nLength Train: {len(sgc_trained)}\nLength Probe: {len(sg_pairs_test)}\nChunk Size: {chunk_size}')
            return [(m,n,p) for m, n, o, p in sg_pairs]
        
        def gen_vc_acquisition():
            # PI+VC session are a total of 16 trials and shuffled as one chunk
            # This is to limit habit development in the rat.
            chunk_size = 4
            # direct route to goal is added for easy checking of repeated routes and then removed  when done creating list
            # LL:0, RR:1, RL:2, LR:3
            # (start arm, cue, route, goal location index)
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((i+1)%4,i,1,i) for i in range(4)] + [((i-1)%4,i,3,i) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [(i, (i-1)%4, 0,i) for i in range(4)] + [((i+2)%4,(i-1)%4, 2,i) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [((i+1)%4,(i+2)%4,1,i) for i in range(4)] + [((i-1)%4,(i+2)%4,3,i) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [(i, (i+1)%4, 0,i) for i in range(4)] + [((i+2)%4,(i+1)%4, 2,i) for i in range(4)]

            sg_pairs_temp = sgc_list

            start_repeat_limit = 3
            goal_repeat_limit = 3
            route_repeat_limit = 3
            start_threepeat = 0
            start_fourpeat = 0
            goal_fourpeat = 0
            goal_threepeat = 0
            route_fourpeat = 0
            route_threepeat = 0
            start_repeat_loc = []
            route_repeat_loc = []
            goal_repeat_loc = []
            sg_pairs = []
            passed = False
            fails = 0
            while not passed:
                passed = True
                sg_pairs.clear()
                for _ in range(chunk_size):
                    random.shuffle(sg_pairs_temp)
                    sg_pairs += sg_pairs_temp.copy()

                # catch fourpeats and locations of threepeats - this is to make sure threepeats aren't
                # too close together.
                for i, sgp in enumerate(sg_pairs[0:-3]):
                    if sgp[2] == sg_pairs[i + 1][2] and sgp[2] == sg_pairs[i + 2][2] and sgp[2] == sg_pairs[i + 3][2]:
                        route_fourpeat += 1
                    if sgp[1] == sg_pairs[i + 1][1] and sgp[1] == sg_pairs[i + 2][1] and sgp[1] == sg_pairs[i + 3][1]:
                        goal_fourpeat += 1
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0] and sgp[0] == sg_pairs[i + 3][0]:
                        start_fourpeat += 1
                    if sgp[2] == sg_pairs[i + 1][2] and sgp[2] == sg_pairs[i + 2][2]:
                        route_threepeat += 1
                        route_repeat_loc.append(i)
                    if sgp[1] == sg_pairs[i + 1][1] and sgp[1] == sg_pairs[i + 2][1]:
                        goal_threepeat += 1
                        goal_repeat_loc.append(i)
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                
                # Catch four repeats or over threepeat limit
                if (route_fourpeat > 0 or goal_fourpeat > 0 or start_fourpeat > 0 or
                        route_threepeat > route_repeat_limit or
                        goal_threepeat > goal_repeat_limit or
                        start_threepeat > start_repeat_limit):
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue

                # Catch route threepeats that are too close together for first occurrence
                if (len(route_repeat_loc) == 3 and (route_repeat_loc[1] - route_repeat_loc[0] < 16) or
                        (len(route_repeat_loc) == 2 and route_repeat_loc[1] - route_repeat_loc[0] < 16)):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(route_repeat_loc) == 3 and (route_repeat_loc[2] - route_repeat_loc[1] < 16):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue

                # Catch goal threepeats that are too close together for first occurance
                if (len(goal_repeat_loc) == 3 and (goal_repeat_loc[1] - goal_repeat_loc[0] < 16) or
                        (len(goal_repeat_loc) == 2 and goal_repeat_loc[1] - goal_repeat_loc[0] < 16)):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(goal_repeat_loc) == 3 and (goal_repeat_loc[2] - goal_repeat_loc[1] < 16):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                
                # Catch route threepeats that are too close together for first occurance
                if (len(start_repeat_loc) == 3 and (start_repeat_loc[1] - start_repeat_loc[0] < 16) or
                        (len(start_repeat_loc) == 2 and start_repeat_loc[1] - start_repeat_loc[0] < 16)):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(start_repeat_loc) == 3 and (start_repeat_loc[2] - start_repeat_loc[1] < 16):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
            
            #print(f'Length List: {len(sg_pairs)}')    
            return [(m,n,p) for m, n, o, p in sg_pairs]
        
        def gen_vc_novel_route_rotate():
            # Novel route probe for VC sessions.
            chunk_size = 3
            # direct route to goal is added for easy checking of repeated routes and then removed  when done creating list
            # LL:0, RR:1, RL:2, LR:3
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((i+1)%4,i,1,i) for i in range(4)] + [((i-1)%4,i,3,i) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [(i, (i-1)%4, 0,i) for i in range(4)] + [((i+2)%4,(i-1)%4, 2,i) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [((i+1)%4,(i+2)%4,1,i) for i in range(4)] + [((i-1)%4,(i+2)%4, 3, i) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [(i, (i+1)%4, 0, i) for i in range(4)] + [((i+2)%4,(i+1)%4, 2, i) for i in range(4)]

            sgc_trained = sgc_list

            sg_pairs_list = []
            # Adding novel route from start arm facing cue
            # LL:0, RR:1, RL:2, LR:3 
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sg_pairs_list = [((i+1)%4, i, 1, i) for i in range(4)] + \
                                [((i-1)%4, i, 3, i) for i in range(4)] + \
                                [((i+2)%4, i%4, 2, i%4) for i in range(16)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sg_pairs_list = [(i, (i-1)%4, 0,i) for i in range(4)] + \
                                [((i+2)%4, (i-1)%4, 2, i) for i in range(4)] + \
                                [((i+1)%4, (i-1)%4, 1, i%4) for i in range(16)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sg_pairs_list = [((i+1)%4, (i+2)%4, 1, i) for i in range(4)] + \
                                [((i-1)%4, (i+2)%4, 3, i) for i in range(4)] + \
                                [(i%4, (i+2)%4, 0, i%4) for i in range(16)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sg_pairs_list = [(i, (i+1)%4, 0, i) for i in range(4)] + \
                                [((i+2)%4, (i+1)%4, 2, i) for i in range(4)] + \
                                [((i-1)%4, (i+1)%4, 3, i%4) for i in range(16)]
            sg_pairs_test = sg_pairs_list
            
            start_repeat_limit = 3
            goal_repeat_limit = 3
            route_repeat_limit = 3
            start_threepeat = 0
            start_fourpeat = 0
            goal_fourpeat = 0
            goal_threepeat = 0
            route_fourpeat = 0
            route_threepeat = 0
            start_repeat_loc = []
            route_repeat_loc = []
            goal_repeat_loc = []
            sg_pairs = []
            temp_item = sg_pairs_test[-1]
            passed = False
            fails = 0
            while not passed:
                passed = True
                sg_pairs.clear()
                for i in range(chunk_size):
                    if i <= 1:
                        random.shuffle(sgc_trained)
                        sg_pairs += sgc_trained.copy()
                    elif i == 2:
                        # Make sure after end of acquisition trial the next trial is a probe
                        sg_pairs_test.remove(temp_item)
                        random.shuffle(sg_pairs_test)
                        sg_pairs += [temp_item] + sg_pairs_test.copy()
                        sg_pairs_test.append(temp_item)
                    else:
                        random.shuffle(sg_pairs_test)
                        if sg_pairs[-6:] == sg_pairs_test:
                            passed = False
                            continue
                        else:
                            sg_pairs += sg_pairs_test.copy()
                # catch fourpeats and locations of threepeats - this is to make sure threepeats aren't
                # too close together.
                for i, sgp in enumerate(sg_pairs[0:-3]):
                    if sgp[2] == sg_pairs[i + 1][2] and sgp[2] == sg_pairs[i + 2][2] and sgp[2] == sg_pairs[i + 3][2]:
                        route_fourpeat += 1
                    if sgp[1] == sg_pairs[i + 1][1] and sgp[1] == sg_pairs[i + 2][1] and sgp[1] == sg_pairs[i + 3][1]:
                        goal_fourpeat += 1
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0] and sgp[0] == sg_pairs[i + 3][0]:
                        start_fourpeat += 1
                    if sgp[2] == sg_pairs[i + 1][2] and sgp[2] == sg_pairs[i + 2][2]:
                        route_threepeat += 1
                        route_repeat_loc.append(i)
                    if sgp[1] == sg_pairs[i + 1][1] and sgp[1] == sg_pairs[i + 2][1]:
                        goal_threepeat += 1
                        goal_repeat_loc.append(i)
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                
                # Catch four repeats or over threepeat limit
                if (route_fourpeat > 0 or goal_fourpeat > 0 or start_fourpeat > 0 or
                        route_threepeat > route_repeat_limit or
                        goal_threepeat > goal_repeat_limit or
                        start_threepeat > start_repeat_limit):
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue

                # Catch route threepeats that are too close together for first occurrence
                if (len(route_repeat_loc) == 3 and (route_repeat_loc[1] - route_repeat_loc[0] < 16) or
                        (len(route_repeat_loc) == 2 and route_repeat_loc[1] - route_repeat_loc[0] < 16)):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(route_repeat_loc) == 3 and (route_repeat_loc[2] - route_repeat_loc[1] < 16):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue

                # Catch goal threepeats that are too close together for first occurance
                if (len(goal_repeat_loc) == 3 and (goal_repeat_loc[1] - goal_repeat_loc[0] < 16) or
                        (len(goal_repeat_loc) == 2 and goal_repeat_loc[1] - goal_repeat_loc[0] < 16)):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(goal_repeat_loc) == 3 and (goal_repeat_loc[2] - goal_repeat_loc[1] < 16):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                
                # Catch route threepeats that are too close together for first occurance
                if (len(start_repeat_loc) == 3 and (start_repeat_loc[1] - start_repeat_loc[0] < 16) or
                        (len(start_repeat_loc) == 2 and start_repeat_loc[1] - start_repeat_loc[0] < 16)):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(start_repeat_loc) == 3 and (start_repeat_loc[2] - start_repeat_loc[1] < 16):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
            
            #print(f'Length Total: {len(sg_pairs)}\nLength Train: {len(sgc_trained)}\nLength Probe: {len(sg_pairs_test)}\nChunk Size: {chunk_size}')
            return [(m,n,p) for m, n, o, p in sg_pairs]

        def gen_vc_reversal_rotate():
            # Reversal probe for VC sessions.
            # Use chunk size to control how many trials there are of reversal (in chunks of 8 trials)
            chunk_size = 10
            # direct route to goal is added for easy checking of repeated routes and then removed  when done creating list
            # LL:0, RR:1, RL:2, LR:3
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sgc_list = [((i+1)%4,i,1,i) for i in range(4)] + [((i-1)%4,i,3,i) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sgc_list = [(i, (i-1)%4, 0,i) for i in range(4)] + [((i+2)%4,(i-1)%4, 2,i) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sgc_list = [((i+1)%4,(i+2)%4,1,i) for i in range(4)] + [((i-1)%4,(i+2)%4, 3, i) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sgc_list = [(i, (i+1)%4, 0, i) for i in range(4)] + [((i+2)%4,(i+1)%4, 2, i) for i in range(4)]
            sgc_trained = sgc_list

            sg_pairs_list = []
            # Adding novel route from start arm facing cue
            # LL:0, RR:1, RL:2, LR:3 
            # (start arm, cue, route, goal location index) 
            if self.agent_cue_goal_orientation == 'N/NE':
                sg_pairs_list = [((i+1)%4, i, 1, (i+2)%4) for i in range(4)] + \
                                [((i-1)%4, i, 3, (i+2)%4) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/SE':
                sg_pairs_list = [(i, (i-1)%4, 0, (i+2)%4) for i in range(4)] + \
                                [((i+2)%4, (i-1)%4, 2, (i+2)%4) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/SW':
                sg_pairs_list = [((i+1)%4, (i+2)%4, 1, (i+2)%4) for i in range(4)] + \
                                [((i-1)%4, (i+2)%4, 3, (i+2)%4) for i in range(4)]
            elif self.agent_cue_goal_orientation == 'N/NW':
                sg_pairs_list = [(i, (i+1)%4, 0, (i+2)%4) for i in range(4)] + \
                                [((i+2)%4, (i+1)%4, 2, (i+2)%4) for i in range(4)]
            sg_pairs_test = sg_pairs_list
            
            start_repeat_limit = 3
            goal_repeat_limit = 3
            route_repeat_limit = 3
            start_threepeat = 0
            start_fourpeat = 0
            goal_fourpeat = 0
            goal_threepeat = 0
            route_fourpeat = 0
            route_threepeat = 0
            start_repeat_loc = []
            route_repeat_loc = []
            goal_repeat_loc = []
            sg_pairs = []
            passed = False
            fails = 0
            while not passed:
                passed = True
                sg_pairs.clear()
                for i in range(chunk_size):
                    if i <= 1:
                        random.shuffle(sgc_trained)
                        sg_pairs += sgc_trained.copy()
                    else:
                        random.shuffle(sg_pairs_test)
                        sg_pairs += sg_pairs_test.copy()
                # catch fourpeats and locations of threepeats - this is to make sure threepeats aren't
                # too close together.
                for i, sgp in enumerate(sg_pairs[0:-3]):
                    if sgp[2] == sg_pairs[i + 1][2] and sgp[2] == sg_pairs[i + 2][2] and sgp[2] == sg_pairs[i + 3][2]:
                        route_fourpeat += 1
                    if sgp[1] == sg_pairs[i + 1][1] and sgp[1] == sg_pairs[i + 2][1] and sgp[1] == sg_pairs[i + 3][1]:
                        goal_fourpeat += 1
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0] and sgp[0] == sg_pairs[i + 3][0]:
                        start_fourpeat += 1
                    if sgp[2] == sg_pairs[i + 1][2] and sgp[2] == sg_pairs[i + 2][2]:
                        route_threepeat += 1
                        route_repeat_loc.append(i)
                    if sgp[1] == sg_pairs[i + 1][1] and sgp[1] == sg_pairs[i + 2][1]:
                        goal_threepeat += 1
                        goal_repeat_loc.append(i)
                    if sgp[0] == sg_pairs[i + 1][0] and sgp[0] == sg_pairs[i + 2][0]:
                        start_threepeat += 1
                        start_repeat_loc.append(i)
                
                # Catch four repeats or over threepeat limit
                if (route_fourpeat > 0 or goal_fourpeat > 0 or start_fourpeat > 0 or
                        route_threepeat > route_repeat_limit or
                        goal_threepeat > goal_repeat_limit or
                        start_threepeat > start_repeat_limit):
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue

                # Catch route threepeats that are too close together for first occurrence
                if (len(route_repeat_loc) == 3 and (route_repeat_loc[1] - route_repeat_loc[0] < 16) or
                        (len(route_repeat_loc) == 2 and route_repeat_loc[1] - route_repeat_loc[0] < 16)):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(route_repeat_loc) == 3 and (route_repeat_loc[2] - route_repeat_loc[1] < 16):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue

                # Catch goal threepeats that are too close together for first occurance
                if (len(goal_repeat_loc) == 3 and (goal_repeat_loc[1] - goal_repeat_loc[0] < 16) or
                        (len(goal_repeat_loc) == 2 and goal_repeat_loc[1] - goal_repeat_loc[0] < 16)):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(goal_repeat_loc) == 3 and (goal_repeat_loc[2] - goal_repeat_loc[1] < 16):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                
                # Catch route threepeats that are too close together for first occurance
                if (len(start_repeat_loc) == 3 and (start_repeat_loc[1] - start_repeat_loc[0] < 16) or
                        (len(start_repeat_loc) == 2 and start_repeat_loc[1] - start_repeat_loc[0] < 16)):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
                if len(start_repeat_loc) == 3 and (start_repeat_loc[2] - start_repeat_loc[1] < 16):
                    sg_pairs = []
                    route_threepeat = 0
                    goal_threepeat = 0
                    start_threepeat = 0
                    start_fourpeat = 0
                    goal_fourpeat = 0
                    route_fourpeat = 0
                    route_repeat_loc = []
                    goal_repeat_loc = []
                    start_repeat_loc = []
                    fails += 1
                    passed = False
                    continue
            
            #print(f"fails: {fails} len: {len(sg_pairs)}")
            #print(f'Length Total: {len(sg_pairs)}\nLength Train: {len(sgc_trained)}\nLength Probe: {len(sg_pairs_test)}\nChunk Size: {chunk_size}')
            return [(m,n,p) for m, n, o, p in sg_pairs]

        #TODO: Test all condition are running the correct amount of trials and go through
        # all cue goal allignments as a file test. f1: is fixed first turn, f2: is fixed second
        # choice in regards to the correct route.
        if self.session_type == 'PI+VC f2 single trial':
            start_goal_cue_list = gen_pi_vc_f2_single_trial()
        elif self.session_type == 'PI+VC f2 acquisition':
            start_goal_cue_list = gen_pi_vc_f2_acq()
        elif self.session_type == 'PI+VC f2 novel route':
            start_goal_cue_list = gen_pi_vc_f2_novel_route()
        elif self.session_type == 'PI+VC f2 no cue':
            start_goal_cue_list = gen_pi_vc_f2_no_cue()
        elif self.session_type == 'PI+VC f2 rotate':
            start_goal_cue_list = gen_pi_vc_f2_rotate()
        elif self.session_type == 'PI+VC f2 reversal':
            start_goal_cue_list = gen_pi_vc_f2_reversal()
        elif self.session_type == 'PI+VC f1 acquisition':
            start_goal_cue_list = gen_pi_vc_f1_acq()
        elif self.session_type == 'PI+VC f1 novel route':
            start_goal_cue_list = gen_pi_vc_f1_novel_route()
        elif self.session_type == 'PI+VC f1 no cue':
            start_goal_cue_list = gen_pi_vc_f1_no_cue()
        elif self.session_type == 'PI+VC f1 rotate':
            start_goal_cue_list = gen_pi_vc_f1_rotate()
        elif self.session_type == 'PI+VC f1 reversal':
            start_goal_cue_list = gen_pi_vc_f1_reversal()
        elif self.session_type == 'PI acquisition':
            start_goal_cue_list = gen_pi_vc_f2_no_cue() # same session design
        elif self.session_type == 'PI novel route no cue':
            start_goal_cue_list = gen_pi_novel_route_no_cue()
        elif self.session_type == 'PI novel route cue':
            start_goal_cue_list = gen_pi_vc_f2_novel_route()
        elif self.session_type == 'PI reversal no cue':
            start_goal_cue_list = gen_pi_reversal_no_cue()
        elif self.session_type == 'PI reversal cue':
            start_goal_cue_list = gen_pi_vc_f2_reversal()
        elif self.session_type == 'VC acquisition':
            start_goal_cue_list = gen_vc_acquisition()
        elif self.session_type == 'VC novel route fixed':
            start_goal_cue_list = gen_pi_vc_f2_novel_route()
        elif self.session_type == 'VC novel route rotate':
            start_goal_cue_list = gen_vc_novel_route_rotate()
        elif self.session_type == 'VC reversal fixed':
            start_goal_cue_list = gen_pi_vc_f2_reversal()
        elif self.session_type == 'VC reversal rotate':
            start_goal_cue_list = gen_vc_reversal_rotate()
        else:
            print('Invalid session type')
            start_goal_cue_list = None

        #print(len(start_goal_cue_list))
        grid_configuration_sequence = []
        len_sgc = len(start_goal_cue_list)
        for i, sgc in enumerate(start_goal_cue_list):
            start_arm = sgc[0]
            cue = sgc[1]
            goal = sgc[2]
            if len_sgc == 1:
                grid_configuration_sequence.append(self.maze_config_trl_list[start_arm][cue][goal])
            elif len_sgc > 1:
                if i < len_sgc - 1:
                    next_start_arm = start_goal_cue_list[i + 1][0]
                grid_configuration_sequence.append(self.maze_config_trl_list[start_arm][cue][goal])
                grid_configuration_sequence.append(self.maze_config_iti_list[next_start_arm])
            else:
                grid_configuration_sequence = None
        
        return grid_configuration_sequence

    def gen_start_pose(self):
        # returns a tuple of the agents starting position and direction
        # check where the start arm is and align the agent so it's pose
        # is facing the center of the maze.
        if self.grid_configuration_sequence[0][13] == 1:
            return rotate_point((6, 8)), rotate_dir(3) # South start arm
        elif self.grid_configuration_sequence[0][14] == 1:
            return rotate_point((4, 6)), rotate_dir(0) # West start arm
        elif self.grid_configuration_sequence[0][15] == 1:
            return rotate_point((6, 4)), rotate_dir(1) # North start arm
        elif self.grid_configuration_sequence[0][16] == 1:
            return rotate_point((8, 6)), rotate_dir(2) # East start arm
        else:
            return DEFAULT_AGENT_START_POS, DEFAULT_AGENT_START_DIR # Default start location

    # State and action supporting functions
    def is_agent_on_obj(self, obj):
        x, y = self.agent_pos
        obj_at_pos = self.grid.get(x, y)
        if isinstance(obj_at_pos, obj):
            return True
        return False

    def gen_obs_grid_mod(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """
        
        topX, topY, botX, botY = self.get_view_exts(agent_view_size)
        
        if self.agent_dir == 0:
            topX -= CELL_VIEW_BEHIND
        elif self.agent_dir == 1:
            topY -= CELL_VIEW_BEHIND
        elif self.agent_dir == 2:
            topX += CELL_VIEW_BEHIND
        elif self.agent_dir == 3:
            topY += CELL_VIEW_BEHIND

        agent_view_size = agent_view_size or self.agent_view_size
        
        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        return grid
    
    def get_pov_render_mod(self, tile_size):
        """
        Render an agent's POV observation as img
        """
        grid = self.gen_obs_grid_mod(AGENT_VIEW_SIZE)

        img = grid.render(
            tile_size,
            agent_pos = self.agent_pov_pos,
            agent_dir=3,
            highlight_mask=None,
        )

        # Make outside grid value black
        img[img == 100] = 0   
        
        # Create Visual Field Mask. This will only see the cue screens and will not see behind the agent
        # in a pyrimindal sweep. This is a simple way to represent the rats visual field.
        visual_condition = (img[:, :, 0] == 255) | (img[:, :, 0] == 25)
        img[:,:,0][~self.visual_mask & visual_condition] = 0

        # Create Wall Ledge Mask.
        img[:, :, 1][self.wall_ledge_mask[:, :, 0]] = 0
        img[:, :, 2][self.wall_ledge_mask[:, :, 1]] = 0

        return img
    
    def get_allocentric_frame(self, tile_size=32):
        # Top-down full map; includes agent if you pass pos/dir
        return self.grid.render(
            tile_size=tile_size,
            agent_pos=self.agent_pos,
            agent_dir=self.agent_dir,
            highlight_mask=None
        )

    def get_trigger_type(self):
        return self.trigger_type

    def plot_observation(self, observation_rgb):
        """
        Plots the full RGB observation along with individual R, G, B channels using matplotlib,
        without creating a new window each time.
        """
        # Check if the figure already exists
        if not hasattr(self, 'fig'):
            # Create the figure and subplots the first time
            self.fig, self.axes = plt.subplots(1, 4, figsize=(16, 4))

        # Clear the content of each axis (without creating a new figure)
        for ax in self.axes:
            ax.clear()

        # Full RGB image
        self.axes[0].imshow(observation_rgb, interpolation='nearest')
        self.axes[0].set_title("Full RGB")
        self.axes[0].axis('off')

        # Separate channels: R, G, B
        channel_titles = ["Cues", "Walls", "Ledges"]
        for i in range(3):  # Iterate over R, G, B channels
            channel_img = observation_rgb[:, :, i]  # Extract the channel
            self.axes[i + 1].imshow(channel_img, cmap="gray", vmin=0, vmax=255, interpolation='nearest')  # Display the channel in grayscale
            self.axes[i + 1].set_title(channel_titles[i])
            self.axes[i + 1].axis('off')

        # Redraw the figure
        plt.draw()
        plt.pause(0.001)  # Small pause to allow the plot to update
    
    def get_action_mask(self):
        """
        Get a mask of the allowed actions based on the current state of the environment
        """
        if self.agent_pose in CORNER_POSES:
            mask = [True, True, True, True]
        else:
            mask = [True, True, True, False]
        return mask

    def get_embeding_obs(self, config, pose):
        x, y, dir = pose
        if [x,y] in [[1,1], [11,1], [11,11], [1,11]]:
            dir = (dir+2)%8
        embedding = []
        return embedding

    # Step function
    def step(self, action):
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False
        # if len(self.grid_configuration_sequence[self.sequence_count]) != 3:
        #     print(self.layout_name_lookup[self.grid_configuration_sequence[self.sequence_count]])
        # else:
        #     for tup in self.grid_configuration_sequence[self.sequence_count]:
        #         print(self.layout_name_lookup[tup])

        # Rotate left
        if action == Actions.left:
            #print('cur pos: ', self.agent_pose, 'last pos: ', self.last_pose)
            # This condition is to have agent move forward weh turning left after exiting a well
            # this is so model moving to the left or right in a single move after exiting a well
            if self.agent_pos in CORNERS and self.last_pose in WELL_EXIT_POSES:
                new_pos_dir = CORNER_LEFT_TURN_FIXES.get(self.agent_pos)
                if new_pos_dir:
                    self.agent_pos, self.agent_dir = new_pos_dir
                reward += FORWARD_SCR
            else:
                self.agent_dir = (self.agent_dir - 1) % 4
                reward += TURN_SCR
            # if self.agent_pos in INTERSECTIONS:
            #     pass
            #     # reward += TURN_INTR_SCR
            # else:
            #     reward += TURN_SCR       
        elif action == Actions.right:
            # Rotate right
            self.agent_dir = (self.agent_dir + 1) % 4
            # if self.agent_pos in INTERSECTIONS:
            #     reward += TURN_INTR_SCR
            # else:
            #     reward += TURN_SCR
            reward += TURN_SCR
        elif action == Actions.forward:
            # Move forward
            if self.agent_pose in WELL_EXIT_POSES:
                # These conditions control well exit behavior
                new_pos_dir = WELL_EXIT_FORWARD_FIXES.get(self.agent_pos)
                if new_pos_dir:
                    self.agent_pos, self.agent_dir = new_pos_dir
                reward += FORWARD_SCR
            elif self.fwd_cell is None or self.fwd_cell.can_overlap():
                self.agent_pos = tuple(self.fwd_pos)
                reward += FORWARD_SCR
            else:
                reward += INAPPROPRIATE_ACTION_SCR # forward movement where it can't happen represents investigation  
        elif action == Actions.pickup:
            # Well entry action (pickup) 
            new_pos_dir = WELL_ENTRY_PICKUP_FIXES.get(self.agent_pos)
            if new_pos_dir:
                self.agent_pos, self.agent_dir = new_pos_dir
            reward += FORWARD_SCR
    
        # Penalize staying in the same position for more that 2 steps
        # This is to prevent the agent from getting stuck in a loop of moving back and forth
        # in the same position or arm
        # if self.agent_pos == self.last_pose[:2]:
        #     self.in_loc_count += 1
        # else:
        #     self.in_loc_count = 0

        # if self.in_loc_count >= 2:
        #     self.in_place_punishment_scr += 1
        # else:
        #     self.in_place_punishment_scr = 0
        
        # Get the cell type the agent is on
        self.cur_cell = type(self.grid.get(*self.agent_pos)).__name__
        # collect last action
        self.last_action = action
        # last pose
        self.last_pose = self.agent_pose
        # Update the agents pose
        self.agent_pose = (*self.agent_pos, self.agent_dir)
        # save agent pose for each step to produce a path history of the session
        self.trajectory.append(self.agent_pose)

        # Capture turn one and turn two scores
        goal_locations = self.grid_configuration_sequence[self.sequence_count][21:25]
        if self.grid_configuration_sequence[self.sequence_count][0] == 2: # Maker this check only happens for a trial not iti.
            ns_goal_active = 1 in (goal_locations[0], goal_locations[2])
            ew_goal_active = 1 in (goal_locations[1], goal_locations[3])
            if self.turn_score[0] == None:
                if ns_goal_active and self.agent_pos in TURN_ONE_NS_MAP:
                    self.turn_score[0] = TURN_ONE_NS_MAP[self.agent_pos]
                elif ew_goal_active and self.agent_pos in TURN_ONE_EW_MAP:
                    self.turn_score[0] = TURN_ONE_EW_MAP[self.agent_pos]
            elif self.turn_score[1] == None:
                if ns_goal_active:
                    if self.agent_pos in TURN_TWO_SET_A:
                        self.turn_score[1] = 1
                    elif self.agent_pos in TURN_TWO_SET_B:
                        self.turn_score[1] = 0
                elif ew_goal_active:
                    if self.agent_pos in TURN_TWO_SET_A:
                        self.turn_score[1] = 0
                    elif self.agent_pos in TURN_TWO_SET_B:
                        self.turn_score[1] = 1

        # CELL AND TRIGGER CONDITION CHECKS
        if self.cur_cell == 'RewardWell':
            reward += WELL_REWARD_SCR
            self.trial_count += 1
            # Single trial condition point variables will reset
            if self.grid_configuration_len == 1:
                terminated = True
                if self.trial_score == None:
                    self.trial_score = 1
                self.episode += 1
                current_episode_data = pd.DataFrame([{'episode': self.episode,
                                                      'turn_scores': self.turn_score,
                                                      'trial_scores': self.trial_score,
                                                      'trajectory': self.trajectory,
                                                      'total_reward': self.session_reward}])
                self.episode_data = pd.concat([self.episode_data, current_episode_data], ignore_index=True)
                self.pseudo_session_score.appendleft(self.trial_score)
                print('Reward: ', self.turn_score, ' ', self.trial_score)
            elif self.grid_configuration_len > 1:
                # multi trial condition with intertrial interval
                if self.trial_score == None:# Checking if this was the first Well visited then set trial score to 1
                        self.trial_score = 1
                # Save trial score and turn scores
                self.episode_trial_scores.append(self.trial_score)
                self.episode_turn_scores.append(self.turn_score)
                print(self.episode_trial_scores)
                # reset trial score, turn scores, step count, and punishment score
                self.trial_score = None # reset trial score
                self.turn_score = [None, None] # reset turn scores
                self.phase_step_count = 0 # reset trial step count
                self.phase_punishment_scr = 0 # reset trial punishment score
                # Checking if last trial and save data and terminate episode if so
                if self.trial_count == self.session_num_trials:
                    self.episode += 1
                    terminated = True
                    self.episode_scores.append(sum(self.episode_trial_scores)/self.session_num_trials)
                    current_episode_data = pd.DataFrame([{'episode': self.episode,
                                                      'turn_scores': self.turn_score,
                                                      'trial_scores': self.trial_score,
                                                      'trajectory': self.trajectory,
                                                      'total_reward': self.session_reward}])
                    self.episode_data = pd.concat([self.episode_data, current_episode_data], ignore_index=True)
                    print('Full Session!',
                          '\n Episode: ', self.episode,
                          '\n Trial Data: ', self.episode_trial_scores, 
                          '\nScore: ', sum(self.episode_trial_scores),
                          '\nNum Trials: ', len(self.episode_trial_scores),
                          '\nAgent Cumulative Rewards: ', self.session_reward)
                else:
                    # Agent in reward well and maze needs to update iti grid configuration
                    # for either a proximal start-arm or a distal start-arm
                    self.sequence_count += 1
                    goal_loc = self.grid_configuration_sequence[self.sequence_count][-1]
                    #print('pre goal: ', goal_loc)
                    # Get the current goal location
                    if self.grid_configuration_sequence[self.sequence_count-1][21] == 1:
                        goal_loc = 0
                    elif self.grid_configuration_sequence[self.sequence_count-1][22] == 1:
                        goal_loc = 1
                    elif self.grid_configuration_sequence[self.sequence_count-1][23] == 1:
                        goal_loc = 2
                    elif self.grid_configuration_sequence[self.sequence_count-1][24] == 1:
                        goal_loc = 3
                    #print('post goal: ', goal_loc)   
                    start_arm_loc = self.grid_configuration_sequence[self.sequence_count+1][-2]
                    #print('pre start: ', start_arm_loc)
                    #print(self.grid_configuration_sequence[self.sequence_count+1])
                    # Get the upcoming start arm location
                    if self.grid_configuration_sequence[self.sequence_count][0][2] == 0:
                        start_arm_loc = 0
                    elif self.grid_configuration_sequence[self.sequence_count][0][5] == 0:
                        start_arm_loc = 1
                    elif self.grid_configuration_sequence[self.sequence_count][0][8] == 0:
                        start_arm_loc = 2
                    elif self.grid_configuration_sequence[self.sequence_count][0][11] == 0:
                        start_arm_loc = 3
                    #print('post start: ', start_arm_loc)
                    # Reminder on session_phase meaning 0: trial, 1: iti_proximal, 2: iti_distal
                    # Make maze adjustments for the iti depending if start arm is in a 
                    # proximal or distal location from the goal.
                    
                    # UPDATE LOCATIONS: Capture cofiguraiton so the observation can change
                    if goal_loc == 0:
                        if start_arm_loc == 0:
                            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][2])
                            self.session_phase = 1
                        elif start_arm_loc == 1:
                            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][1])
                            self.session_phase = 1
                        else:    
                            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][0])
                            self.session_phase = 2
                    elif goal_loc == 1:
                        if start_arm_loc == 1:
                            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][2])
                            self.session_phase = 1
                        elif start_arm_loc == 2:
                            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][1])
                            self.session_phase = 1
                        else:
                            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][0])
                            self.session_phase = 2  
                    elif goal_loc == 2:
                        if start_arm_loc == 2:
                            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][2])
                            self.session_phase = 1
                        elif start_arm_loc == 3:
                            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][1])
                            self.session_phase = 1
                        else:
                            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][0])
                            self.session_phase = 2  
                    elif goal_loc == 3: 
                        if start_arm_loc == 3:
                            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][2])
                            self.session_phase = 1
                        elif start_arm_loc == 0:
                            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][1])
                            self.session_phase = 1
                        else:
                            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][0])
                            self.session_phase = 2

        elif self.cur_cell == 'EmptyWell' and self.last_pose[:2] not in WELL_LOCATIONS:
            reward += WELL_EMPTY_SCR
            self.trial_score = 0
        elif (self.is_agent_on_obj(Trigger)):
            self.trial_score = None
            self.turn_score = [None, None]
            goal_loc = self.grid_configuration_sequence[self.sequence_count][-1]
            if self.grid.get(*self.agent_pos).get_trigger_type() == 'A':
                self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][1])
            elif self.grid.get(*self.agent_pos).get_trigger_type() == 'B':
                self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][2])
            elif self.grid.get(*self.agent_pos).get_trigger_type() == 'S':
                self.sequence_count += 1
                self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count])
                self.session_phase = 0
                self.phase_step_count = 0 # reset trial step count
                self.phase_punishment_scr = 0 # reset phase punishment score
                

        # Update fwd position after possible maze changes
        # Get the position in front of the agent
        self.fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        self.fwd_cell = self.grid.get(*self.fwd_pos)

        # Check for too long in segment penalty and give penalty if step for phase exceeded
        self.phase_step_count += 1
        if self.session_phase == 0 and self.phase_step_count > 13:
            self.phase_punishment_scr += TOO_LONG_IN_PHASE
            reward += self.phase_punishment_scr
        elif self.session_phase == 1 and self.phase_step_count > 10:
            self.phase_punishment_scr += TOO_LONG_IN_PHASE
            reward += self.phase_punishment_scr
        elif self.session_phase ==2 and self.phase_step_count > 19:
            self.phase_punishment_scr += TOO_LONG_IN_PHASE
            reward += self.phase_punishment_scr

        # if self.phase_step_count >= TOO_LONG_SCR:
        #     self.phase_punishment_scr += TOO_LONG_CONST
        #     reward += self.phase_punishment_scr
        
        if self.render_mode == "human":
            self.render()


        # Add in in-place punishment score
        # reward -= 0.05 * self.in_place_punishment_scr

        # Track session cumulative reward
        self.session_reward += reward

        if self.step_count >= self.max_steps:
            if self.grid_configuration_len == 1:
                reward += TIME_OUT_SCR
                print('DNF: ', self.turn_score, ' ', self.trial_score)
                self.episode += 1
                self.trial_score = 0
                current_episode_data = pd.DataFrame([{'episode': self.episode,
                                                      'turn_scores': self.turn_score,
                                                      'trial_scores': self.trial_score,
                                                      'trajectory': self.trajectory,
                                                      'total_reward': self.session_reward}])
                self.episode_data = pd.concat([self.episode_data, current_episode_data], ignore_index=True)
                self.pseudo_session_score.appendleft(self.trial_score)
                truncated = True    
            elif self.grid_configuration_len > 1:
                reward += TIME_OUT_SCR
                self.episode += 1
                truncated = True
                if self.trial_score == None:
                    self.trial_score = 0
                self.episode_trial_scores.append(self.trial_score)
                self.episode_turn_scores.append(self.turn_score)
                self.episode_scores.append(sum(self.episode_trial_scores)/self.session_num_trials)
                current_episode_data = pd.DataFrame([{'episode': self.episode,
                                                      'turn_scores': self.turn_score,
                                                      'trial_scores': self.trial_score,
                                                      'trajectory': self.trajectory,
                                                      'total_reward': self.session_reward}])
                self.episode_data = pd.concat([self.episode_data, current_episode_data], ignore_index=True)
                print('DNF!',
                      '\n Episode: ', self.episode,
                      '\n Trial Data: ', self.episode_trial_scores, 
                      '\nScore: ', sum(self.episode_trial_scores),
                      '\nNum Trials: ', len(self.episode_trial_scores),
                      '\nAgent Cumulative Rewards: ', self.session_reward)
                
        print(self.agent_pose)
        #obs = self.gen_obs()
        img = self.get_pov_render_mod(tile_size=VIEW_TILE_SIZE)
        #img = self.get_allocentric_frame(tile_size=VIEW_TILE_SIZE)
        #self.plot_observation(img)
        obs_mod = {'image': img,
                   'direction': self.agent_dir,
                   'mission': self.mission,}
        # action mask
        info = {'action_mask': self.get_action_mask(), 
                'agent_pos': self.agent_pos,
                'terminated': terminated, 
                'truncated': truncated,
                'episode_scores': self.episode_scores,
                'session_reward': self.session_reward}
        #print('Step Reward: ', reward, '\nTotal Reward: ', self.session_reward)
        #print('Session Phase: ', self.session_phase, 'Phase Step: ', self.phase_step_count)
        return obs_mod, reward, terminated, truncated, info

    @staticmethod
    def _gen_mission():
        return "corner maze mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid =  Grid(width, height)

        # build basic maze structure
        # reset maze state array
        self.maze_state_array = [0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
          
        # Layout entire maze as Chasm objects
        self.put_obj_rect(Chasm(), 0, 0, width, height)
        # Place empty grid elements that agent can move on
        for i in range(3):
            for j in range(9):
                self.grid.set(j+2, 4*i+2, None)
        for i in range(3):
            for j in range(3):
                self.grid.set(4*i+2, j+3, None)
                self.grid.set(4*i+2, j+7, None)

        # Make Displays with cues off
        self.put_obj(Wall(color='cue_off_rgb'), 6,1)
        self.put_obj(Wall(color='cue_off_rgb'), 11, 6)
        self.put_obj(Wall(color='cue_off_rgb'), 6, 11)
        self.put_obj(Wall(color='cue_off_rgb'), 1, 6)
        # Place wells
        self.put_obj(EmptyWell(), 11, 1)
        self.put_obj(EmptyWell(), 11, 11)
        self.put_obj(EmptyWell(), 1, 11)
        self.put_obj(EmptyWell(), 1, 1)

        # Build session configuration sequence data
        self.grid_configuration_sequence = self.gen_grid_configuration_sequence()
        self.grid_configuration_len = len(self.grid_configuration_sequence)
        self.session_num_trials = -(-self.grid_configuration_len // 2) # ceiling division trick to get number of trials

        # Configure maze environment to the first setting of grid_configuration_sequence
        self.update_grid_configuration(self.grid_configuration_sequence[0])

        # Determine start position from grid config and set agent start position and direction
        self.agent_pos, self.agent_dir = self.gen_start_pose()
        self.agent_pose = (*self.agent_pos, self.agent_dir)
        self.agent_start_pos = self.agent_pos
        self.session_phase = 0 #agent starts session in trial phase

        # Update position
        # Get the position in front of the agent
        self.fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        self.fwd_cell = self.grid.get(*self.fwd_pos)
        # Get the cell type the agent is on
        self.cur_cell = type(self.grid.get(*self.agent_pos)).__name__
        # save agent pose for each step to produce a path history of the session
        self.trajectory.append(self.agent_pose)

        self.mission = "corner maze mission"

# REGION ######################### BEGIN MANUAL CONTROL CODE ###############################
class ExtendedManualControl(ManualControl):
    def __init__(self, env: CornerMazeEnv,  seed=None) -> None:
        self.env = env
        self.seed = seed
        self.closed = False

    def key_handler(self, event):
        # Ensure available_actions is updated dynamically
        key: str = event.key
        if self.env.key_actions == 0:
            key_to_action = {
                "left": Actions.left,
                "right": Actions.right,
                "up": Actions.forward
            }
        elif self.env.key_actions == 1: # agent on well facing casm so can toggle and enter reward arm to consume reward
            key_to_action = {
                "left": Actions.left,
                "right": Actions.right,
                "up": Actions.forward,
                "space": Actions.toggle
            }
        elif self.env.key_actions == 2: # post toggle action for left turn
            key_to_action = {
                "left": Actions.left,
                "up": Actions.forward,
                "space": Actions.toggle,
                "down": Actions.pickup
            }
        elif self.env.key_actions == 3: # post toggle action for right turn
            key_to_action = {
                "right": Actions.right,
                "up": Actions.forward,
                "space": Actions.toggle,
                "down": Actions.pickup
            }
        else:
            key_to_action = {
                "left": Actions.left,
                "right": Actions.right,
                "up": Actions.forward
            }

        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print('invalid key')

# REGION ######################### BEGIN RL AGENT CODE ###############################
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 32, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (3, 3)),
            nn.ReLU(),
#            nn.Conv2d(16, 32, (3, 3)),
#            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
   
class SaveEnvDataFrameCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveEnvDataFrameCallback, self).__init__(verbose)
        self.save_path = save_path

    def _on_step(self):
        infos = self.locals.get('infos', None)
        if infos is not None:
            # Iterate over each environment's info dictionary
            for env_idx, info in enumerate(infos):
                # Retrieve the variable from the info dictionary
                terminated = info.get('terminated', None)
                episode_scores = info.get('episode_scores', None)
                episode_reward = info.get('session_reward', None)

        if (terminated == True
            and len(episode_scores) > 1
            and episode_scores[-1] >= 0.72 
            and episode_scores[-2] >= 0.72
            #and episode_reward > 8
            ):
            
            # Check if the cumulative score has reached the threshold
            print(f"Agent trained. Stopping training.")
            return False  # Returning False stops training
        return True

    def _on_training_end(self):
        # Unwrap environment to access episode_data
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
            env = env.env  # Unwrap all layers

        # Access and save the DataFrame
        df = env.episode_data
        df.to_csv("dataframe_output.csv", index=False)
        df.to_parquet(self.save_path, engine="pyarrow")
        print(f"DataFrame saved to {self.save_path}")

class CustomMaskablePolicy(MaskableActorCriticPolicy):

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        This addition attempts to modify logits based on the agent's position
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :param action_masks: Action masks to apply to the action distribution
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        # Inject logit modification here
        action_logits = self.action_net(latent_pi)
        # Access agent_pos from the `info` dictionary
        infos = self.locals.get("infos", [{}])
        agent_pos = infos[0].get("agent_pos", None)  # Single environment case
        # Modify the logits based on agent_pos
        if agent_pos is not None:
            action_logits = self.modify_logits_based_on_state(action_logits, agent_pos)
        distribution = self._get_action_dist_from_latent(action_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def modify_logits_based_on_state(self, logits, agent_pos):
        # Example: Adjust logits based on agent's position
        if agent_pos not in INTERSECTIONS:
            logits[0] *= LOGIT_TURN_SCALER
            logits[1] *= LOGIT_TURN_SCALER  # Example adjustment
        return logits

# REGION ######################### MAIN BLOCK ###############################
def main():
    # run_mode 0: manual control testing with single trial and pov plot
    # run_mode 1: manual control with single trial and view of agent
    # run_mode 2: manual control testing with full session (enter session_type you want to use)
    # run_mode 3: PPO RL model with single trial (train) No masking.
    # run_mode 4: maskablePPO RL model with full session (enter session_type you want to train on)
    # run_mode 5: A2C RL model with single trial (train) No masking.
    # run_mode 6: Run trained RL model in inference mode
    # run_mode 7: Run trained RL model on novel route and continue training
    mode = 2
    
    # Define the MiniGrid action legend
    action_legend = {
        0: "L",
        1: "R",
        2: "F",
        3: "P",
        4: "D",
        5: "T",
        6: "D"
    }
    # Session_types: 
    # 'PI+VC f2 acquisition', 'PI+VC f2 novel route', 'PI+VC f2 rotate', 'PI+VC f2 no cue', 'PI+VC f2 reversal'
    # 'PI+VC f1 acquisition', 'PI+VC f1 novel route', 'PI+VC f1 rotate', 'PI+VC f1 no cue', 'PI+VC f1 reversal'
    # 'PI acquisition', 'PI novel route cue', 'PI novel route no cue', 'PI reversal cue', 'PI reversal no cue'
    # 'VC acquisition' 'VC novel route fixed', 'VC novel route rotate', 'VC reversal fixed', 'VC reversal rotate'
    if mode == 0:
        env = CornerMazeEnv(render_mode="human",
                            max_steps=10000,
                            agent_cue_goal_orientation='N/NE',
                            start_goal_location = 'NE',
                            session_type='PI+VC single trial',
                            run_mode=mode)
        
        manual_control = ManualControl(env, seed=42)
        manual_control.start()
    elif mode == 1:
        env = CornerMazeEnv(render_mode="human",
                            max_steps=10000,
                            agent_cue_goal_orientation='N/NE',
                            start_goal_location = 'NE',
                            session_type='PI+VC single trial',
                            run_mode=mode)
        manual_control = ManualControl(env, seed=42)
        manual_control.start()
    elif mode == 2:
        env = CornerMazeEnv(render_mode="human",
                            max_steps=10000,
                            agent_cue_goal_orientation='N/NE',
                            start_goal_location = 'NE',
                            session_type='PI+VC f2 acquisition',
                            run_mode=mode)
        manual_control = ManualControl(env, seed=42)
        manual_control.start()
    elif mode == 3:
        MODEL = 'PPO'
        STEPS = 900
        N_STEPS = 64
        BATCH_SIZE = 8
        for i in range(1):
            MODEL_SEED = random.randint(100000, 999999)
            policy_kwargs = dict(
                features_extractor_class=MinigridFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=128),
            )
            #env = CornerMazeEnv(render_mode="human",
            env = CornerMazeEnv(render_mode="human",
                                max_steps=STEPS,
                                agent_cue_goal_orientation='N/NE',
                                start_goal_location = 'N',
                                session_type='VC reversal rotate',
                                run_mode=mode)
            env = ImgObsWrapper(env)
            model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, 
                        n_steps=N_STEPS, 
                        batch_size=BATCH_SIZE, 
                        seed=MODEL_SEED)
            # Initialize the callback with the specified report interval and verbose level
            #report_callback = CustomLoggingCallback(report_interval=100, print_actions=True,  action_legend=action_legend, verbose=
            # Specify the path to save the DataFrame as a Parquet file
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            trial_directory = f"{MODEL}-{STEPS}-{N_STEPS}-{BATCH_SIZE}-MagicWall"
            os.makedirs(f"model-data/{trial_directory}/trial_{timestamp}", exist_ok=True)
            path_1 = f"model-data/{trial_directory}/trial_{timestamp}/trial_data_env_{MODEL_SEED}_{STEPS}_{N_STEPS}_{BATCH_SIZE}_{timestamp}.parquet"
            path_2 = f"model-data/{trial_directory}/trial_{timestamp}/trial_data_sb3_{MODEL_SEED}_{STEPS}_{N_STEPS}_{BATCH_SIZE}_{timestamp}"
            print(f"Saving to path: {path_1}")
            callback = SaveEnvDataFrameCallback(save_path=path_1)
            # Train the model with the custom callback
            model.learn(total_timesteps=1e5, callback=callback)
            model.save(path_2)
    elif mode == 4:
        MODEL = 'PPO'
        N_STEPS = 64
        STEPS = 1536 * 4
        BATCH_SIZE = 16
        for i in range(1):
            MODEL_SEED = random.randint(100000, 999999)
            policy_kwargs = dict(
                features_extractor_class=MinigridFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=128),
            )
            #env = CornerMazeEnv(render_mode="human",
            env = CornerMazeEnv(render_mode="rgb_array",
                                max_steps=STEPS,
                                agent_cue_goal_orientation='N/NE',
                                start_goal_location = 'NE',
                                session_type='PI+VC acquisition',
                                run_mode=mode)
            
            env = ImgObsWrapper(env)
            def mask_fn(env):
                return env.unwrapped.get_action_mask()
            env = ActionMasker(env, mask_fn)

            model = MaskablePPO("CnnPolicy",
                                env,
                                n_steps=N_STEPS,
                                batch_size=BATCH_SIZE,
                                n_epochs=4,               # Fewer epochs = faster updates, still robust
                                gamma=0.999,              # High to prioritize long-term return
                                gae_lambda=0.95,          # Standard for GAE (generalized Advantage Estimation)
                                learning_rate=1e-5,       # Higher for sparse reward envs
                                ent_coef=0.01,            # Encourages exploration
                                vf_coef=0.5,              # Typical value
                                clip_range=0.2,             # PPO clipping range
                                max_grad_norm=0.5,         # Clips gradients to prevent exploding gradients
                                policy_kwargs=policy_kwargs, 
                                verbose=0,    
                                seed=MODEL_SEED)

            # Initialize the callback with the specified report interval and verbose level
            #report_callback = CustomLoggingCallback(report_interval=100, print_actions=True,  action_legend=action_legend, verbose=

            # Specify the path to save the DataFrame as a Parquet file
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            trial_directory = f"{MODEL}-{STEPS}-{N_STEPS}-{BATCH_SIZE}-MagicWall-Allocentric"

            os.makedirs(f"model-data/{trial_directory}/trial_{timestamp}", exist_ok=True)
            path_1 = f"model-data/{trial_directory}/trial_{timestamp}/trial_data_env_{MODEL_SEED}_{STEPS}_{N_STEPS}_{BATCH_SIZE}_{timestamp}.parquet"
            path_2 = f"model-data/{trial_directory}/trial_{timestamp}/trial_data_sb3_{MODEL_SEED}_{STEPS}_{N_STEPS}_{BATCH_SIZE}_{timestamp}"
            print(f"Saving to path: {path_1}")
            callback = SaveEnvDataFrameCallback(save_path=path_1)

            # Train the model with the custom callback
            model.learn(total_timesteps=1e7, callback=callback)
            model.save(path_2)
    elif mode == 5:
        policy_kwargs = dict(
            features_extractor_class=MinigridFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )
        #env = CornerMazeEnv(render_mode="human",
        env = CornerMazeEnv(
                            max_steps=100,
                            agent_cue_goal_orientation='N/NE',
                            start_goal_location = 'NE',
                            session_type='PI+VC single trial',
                            run_mode=mode)
        env = ImgObsWrapper(env)
        
        policy_kwargs = dict(
            features_extractor_class=MinigridFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=512),
        )

        model = A2C("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
        
        # Start training with the custom callback
        model.learn(50000)
        model.save("a2c_trained_model")
    elif mode == 6:
        # Load the trained model
        model = MaskablePPO.load("model-data/PPO-6144-64-16-MagicWall/trial_20250806101148/trial_data_sb3_898991_6144_64_16_20250806101148.zip")
        #model = MaskablePPO.load("model-data/PPO-6144-64-16-MagicWall/trial_20250805140536/trial_data_sb3_790972_6144_64_16_rewward_greater_0_20250805140536.zip")
        #model = MaskablePPO.load("model-data/PPO-6144-64-16-MagicWall/trial_20250805165515/trial_data_sb3_290580_6144_64_16_greater_20_20250805165515.zip")
        #model = MaskablePPO.load("model-data/PPO-6144-64-16-MagicWall/trial_20250807180546/trial_data_sb3_372033_6144_64_16_20250807180546.zip")
        # Create the environment
        env = CornerMazeEnv(render_mode="human",
                            max_steps=100000,
                            agent_cue_goal_orientation='N/NE',
                            start_goal_location = 'NE',
                            session_type='PI+VC reversal',
                            run_mode=mode)
        env = ImgObsWrapper(env)
        def mask_fn(env):
            return env.unwrapped.get_action_mask()
        env = ActionMasker(env, mask_fn)
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True, action_masks=info['action_mask'])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()
    elif mode == 7:
        MODEL = 'PPO'
        N_STEPS = 64
        STEPS = 1536 * 4
        BATCH_SIZE = 16
        for i in range(1):
            MODEL_SEED = random.randint(100000, 999999)
            policy_kwargs = dict(
                features_extractor_class=MinigridFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=128),
            )
            env = CornerMazeEnv(render_mode="human",
            #env = CornerMazeEnv(render_mode="rgb_array",
                                max_steps=STEPS,
                                agent_cue_goal_orientation='N/NE',
                                start_goal_location = 'random',
                                session_type='PI acquisition',
                                run_mode=mode)
            
            env = ImgObsWrapper(env)
            def mask_fn(env):
                return env.unwrapped.get_action_mask()
            env = ActionMasker(env, mask_fn)

            #model = MaskablePPO.load("model-data/PPO-6144-64-16-MagicWall/trial_20250805140536/trial_data_sb3_790972_6144_64_16_rewward_greater_0_20250805140536.zip",
            #                         env=env)
            model = MaskablePPO.load("model-data/PPO-6144-64-16-MagicWall/trial_20250807180546/trial_data_sb3_372033_6144_64_16_20250807180546.zip",
                                     env=env)
            # Train the model with the custom callback
            model.learn(total_timesteps=10e6)

if __name__ == "__main__":
    main()
