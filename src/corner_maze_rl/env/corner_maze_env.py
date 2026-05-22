from __future__ import annotations

from typing import Any

from collections import deque
from itertools import combinations

import numpy as np
import random
from .constants import (
    ACQUISITION_SESSION_TRIALS,
    ACTION_ENTER_WELL, ACTION_PAUSE,
    CRITERION_CONSECUTIVE_REQUIRED, CRITERION_SCORE_THRESHOLD,
    AGENT_VIEW_BEHIND, AGENT_VIEW_SIZE,
    BARRIER_LOCATIONS, CELL_VIEW_BEHIND,
    CORNER_POSES, CORNERS, CUE_LOCATIONS,
    EMBEDDING_DIM, EMBEDDING_PARQUET_PATH, EYE_IMG_SIZE,
    STEREO_IMG_SIZE, STEREO_DATASET_PATH,
    VIEW_PARQUET_PATH,
    EXPB_ACCLIMATION_STEPS, EXPB_BARRIER_DELAY_STEPS,
    EXPB_MAX_STEPS, EXPB_NUM_REWARDS,
    EXPOSURE_ITI_STEPS, EXPOSURE_ITI_STD,
    EXPOSURE_MAX_STEPS, EXPOSURE_NUM_REWARDS,
    GOAL_LOCATION_MAP, NUM_ARMS,
    PRETRIAL_MIN_STEPS, PRETRIAL_TRIGGER_POSITIONS,
    RENDER_FPS,
    STATE_BASE, STATE_EXPA, STATE_EXPB, STATE_PRETRIAL, STATE_TRIAL, STATE_ITI,
    STEP_FORWARD_COST, STEP_TURN_COST,
    TRIGGER_LOCATIONS, VIEW_TILE_SIZE,
    WELL_EXIT_POSES, WELL_LOCATIONS, WELL_REWARD_SCR,
)
from .trial_sequence_validation import (
    validate_sequence_start_only, validate_sequence_multi,
    shuffle_uniform_chunks, shuffle_acq_then_probe, shuffle_acq_then_novel,
)
from .trial_sequence_gen import (
    get_f2_trained_pairs, get_f2_no_cue_pairs,
    get_f2_novel_route_probe_pairs, get_f2_reversal_probe_pairs,
    get_f2_rotate_pairs,
    get_f1_trained_pairs, get_f1_no_cue_pairs,
    get_f1_novel_route_probe_pairs, get_f1_reversal_probe_pairs,
    get_f1_rotate_pairs,
    get_pi_novel_route_no_cue_probe_pairs, get_pi_reversal_no_cue_probe_pairs,
    get_vc_novel_route_rotate_probe_pairs, get_vc_reversal_rotate_probe_pairs,
)

from gymnasium import spaces
from gymnasium.core import ObsType

from minigrid.core.constants import COLORS, COLOR_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Ball, Floor
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_rect

import pandas as pd

# CONDITIONAL LOOKUP DICTIONARIES
DEFAULT_AGENT_START_POS = (11, 1)
DEFAULT_AGENT_START_DIR = 1

CORNER_LEFT_TURN_WELL_EXIT = {
    (10, 2): ((10, 3), 1),
    (10, 10): ((9, 10), 2),
    (2, 10): ((2, 9), 3),
    (2, 2): ((3, 2), 0),
}

WELL_EXIT_FORWARD = {
    (11, 1): ((10, 2), 2),
    (11, 11): ((10, 10), 3),
    (1, 11): ((2, 10), 0),
    (1, 1): ((2, 2), 1),
}

WELL_ENTRY_PICKUP = {
    (10, 10): ((11, 11), 1),
    (2, 10): ((1, 11), 2),
    (2, 2): ((1, 1), 3),
    (10, 2): ((11, 1), 0),
}

TURN_ONE_SENW_MAP = {
    (7, 6): 1,
    (5, 6): 0,
    (6, 5): 1,
    (6, 7): 0,
}

TURN_ONE_NESW_MAP = {
    (7, 6): 0,
    (5, 6): 1,
    (6, 5): 0,
    (6, 7): 1,
}

TURN_TWO_SET_A = {(10, 7), (2, 5), (5, 2), (7, 10)}
TURN_TWO_SET_B = {(10, 5), (2, 7), (7, 2), (5, 10)}

# Exposure B barrier progression sequence
# (layout_name, trigger_type, trigger_pos, delay_after)
EXPB_BARRIER_SEQUENCE = [
    ('expb_xxxx_x_xx', 'timed', None,   EXPB_ACCLIMATION_STEPS),  # step 0: acclimation (agent trapped at center pocket)
    ('expb_exxx_x_xx', 'timed', None,   EXPB_BARRIER_DELAY_STEPS),  # step 1: CE drops after acclimation
    ('expb_enxx_x_xx', 'zone',  (8, 6), EXPB_BARRIER_DELAY_STEPS),  # step 2: CN drops at (8,6)
    ('expb_enwx_x_xx', 'zone',  (6, 4), EXPB_BARRIER_DELAY_STEPS),  # step 3: CW drops at (6,4)
    ('expb_enws_x_xx', 'zone',  (4, 6), EXPB_BARRIER_DELAY_STEPS),  # step 4: CS drops at (4,6)
    ('expb_x_x_xx',    'zone',  (6, 8), 0),  # step 5: arm barriers drop at (6,8) → Phase B (fully open, ≡ expa)
]

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

class Well(Ball):
    def __init__(self, color='black', visible=False, has_reward=False):
        super().__init__(color)
        self.visible = visible
        self.has_reward = has_reward

    def can_toggle(self):
        return True

    def can_overlap(self):
        return True

    def render(self, img):
        if self.visible:
            c = COLORS[self.color]
            fill_coords(img, point_in_rect(0, 1, 0, 1), c)

class Trigger(Floor):
    def __init__(self, color='black', visible=False, trigger_type=None):
        super().__init__(color)
        self.visible = visible
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
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}
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
        obs_mode: str = "view",
        criterion_threshold: float = CRITERION_SCORE_THRESHOLD,
        criterion_consecutive: int = CRITERION_CONSECUTIVE_REQUIRED,
        trial_configs: list | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            if session_type == 'exposure':
                max_steps = EXPOSURE_MAX_STEPS
            elif session_type == 'exposure_b':
                max_steps = EXPB_MAX_STEPS
            else:
                max_steps = 4 * size**2

        # Set added initialization variables
        self.session_type = session_type
        self.agent_cue_goal_orientation = agent_cue_goal_orientation
        self.start_goal_location = start_goal_location
        # When supplied, gen_grid_configuration_sequence skips the random
        # SESSION_GENERATORS shuffle and uses this list verbatim. Used by
        # yoked replay so the env's start arm / cue / goal sequence
        # matches the rat's actual recorded trials.
        self._fixed_trial_configs = trial_configs

        # Observation mode: "view" (21x21x3 RGB), "embedding" (60D vector),
        # or "stereo" (96x96x2 stereo eye images)
        if obs_mode not in ("view", "embedding", "stereo"):
            raise ValueError(f"obs_mode must be 'view', 'embedding', or 'stereo', got '{obs_mode}'")
        self.obs_mode = obs_mode

        # Training criterion configuration (stored before super().__init__
        # which triggers init_variables())
        self.criterion_threshold = criterion_threshold
        self.criterion_consecutive = criterion_consecutive

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        # Define the observation space
        if self.obs_mode == "view":
            self.observation_space = spaces.Dict({
                "image": spaces.Box(
                    low=0, high=255, shape=(AGENT_VIEW_SIZE, AGENT_VIEW_SIZE, 3), dtype=np.uint8
                ),
                "direction": spaces.Discrete(4),
                "mission": mission_space,
            })
        elif self.obs_mode == "embedding":
            self.observation_space = spaces.Dict({
                "embedding": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(EMBEDDING_DIM,), dtype=np.float64
                ),
                "direction": spaces.Discrete(4),
                "mission": mission_space,
            })
        else:  # stereo
            self.observation_space = spaces.Dict({
                "stereo": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(STEREO_IMG_SIZE, STEREO_IMG_SIZE, 2),
                    dtype=np.float32,
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
        # state_type: [0] STATE_BASE/EXPA/EXPB/PRETRIAL/TRIAL/ITI
        # maze_state_array: [1-16] = barriers, [17-20] = cues, [21-24] = wells,
        # [25-36] = trigger zones. reference spread sheet for more details
        self.maze_state_array = [STATE_BASE, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        
        # Dictionary to store trl layouts
        self.layouts = {}

        # Initial layout configuration, no barriers, no cues, no rewards, no triggers
        # Layout dynamic naming: layout_phase_{start_arm}_{cue}_{goal}
        self.layouts['x_x_xx'] = [STATE_BASE, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        
        # Exposure A configurations: open maze with reward wells, no barriers, no cues, no triggers
        # Naming: expa_x_x_{active wells} — all non-empty subsets of {se, sw, nw, ne}
        well_names = ['se', 'sw', 'nw', 'ne']  # matches WELL_LOCATIONS index order
        for r in range(1, 5):
            for combo in combinations(range(4), r):
                name = 'expa_x_x_' + '_'.join(well_names[i] for i in combo)
                layout = [STATE_EXPA, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
                for i in combo:
                    layout[21 + i] = 1
                self.layouts[name] = layout

        # Exposure B configurations: progressive barrier lowering, no wells, no cues
        # 4-letter slot = which CENTER barriers have dropped (e=CE, n=CN, w=CW, s=CS).
        # 'xxxx' (all wildcards) = acclimation state, all 4 center + 4 arm barriers up
        #     (agent spawns at (6,6) and is trapped in the center pocket).
        # Letters accumulate as barriers drop in the session order (see EXPB_BARRIER_SEQUENCE).
        # 'x_x_xx' (collapsed slot) = Phase B end-state, arm barriers dropped too — fully
        #     open and visually identical to expa_x_x_xx (see _get_pose_label).
        # Center barriers: idx 13=(7,6) CE, idx 14=(6,7) CS, idx 15=(5,6) CW, idx 16=(6,5) CN
        # Arm-adj-to-center barriers: idx 2=(9,6) E, idx 5=(6,9) S, idx 8=(3,6) W, idx 11=(6,3) N
        #                              st  E-arm    S-arm    W-arm    N-arm    Center   cues     wells
        self.layouts['expb_xxxx_x_xx'] = [STATE_EXPB, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 1,1,1,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        self.layouts['expb_exxx_x_xx'] = [STATE_EXPB, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,1,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        self.layouts['expb_enxx_x_xx'] = [STATE_EXPB, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        self.layouts['expb_enwx_x_xx'] = [STATE_EXPB, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        self.layouts['expb_enws_x_xx'] = [STATE_EXPB, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        self.layouts['expb_x_x_xx']    = [STATE_EXPB, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]

        # Define dimensions for start arm, cue, and goal
        start_arms = ['n', 'e', 's', 'w']
        cues = ['n', 'e', 's', 'w', 'x']
        goals = ['ne', 'se', 'sw', 'nw']

        # Dynamically build layout variables for all trial configurations
        # This follows maze layouts as defined in the 2S2C behavioral task
        # variable naming: trl_{start_arm}_{cue}_{goal}
        base_trl_layouts = {
            'n' : [STATE_TRIAL, 0,0,0, 1,0,1, 0,0,0, 1,0,1, 0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0],
            'e' : [STATE_TRIAL, 1,0,1, 0,0,0, 1,0,1, 0,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0],
            's' : [STATE_TRIAL, 0,0,0, 1,0,1, 0,0,0, 1,0,1, 0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0],
            'w' : [STATE_TRIAL, 1,0,1, 0,0,0, 1,0,1, 0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,0]
        }
        for start_arm in start_arms:
            for cue in cues:
                for goal in goals:
                    variable_name = f'trl_{start_arm}_{cue}_{goal}'
                    layout = base_trl_layouts[start_arm].copy()
                    layout[17] = 1 if cue == 'e' else 0   # CUE_LOCATIONS[0]=(11,6) = east
                    layout[18] = 1 if cue == 's' else 0   # CUE_LOCATIONS[1]=(6,11) = south
                    layout[19] = 1 if cue == 'w' else 0   # CUE_LOCATIONS[2]=(1,6)  = west
                    layout[20] = 1 if cue == 'n' else 0   # CUE_LOCATIONS[3]=(6,1)  = north
                    layout[21] = 1 if goal == 'se' else 0  # WELL_LOCATIONS[0]=(11,11) = SE
                    layout[22] = 1 if goal == 'sw' else 0  # WELL_LOCATIONS[1]=(1,11)  = SW
                    layout[23] = 1 if goal == 'nw' else 0  # WELL_LOCATIONS[2]=(1,1)   = NW
                    layout[24] = 1 if goal == 'ne' else 0  # WELL_LOCATIONS[3]=(11,1)  = NE

                    self.layouts[variable_name] = layout

        # ITI Configurations: location of start arm is stated after iti
        # the meaning of the goal location is different here it indicates the type of ITI
        # such that the maze is configured to lead to the next start arm location while including that well location.
        # when the goal is xx it means two wells are present to enter but the rat must go to the far side of the maze to get to the
        # start arm. In all ITI configurations the well is in the empty state.
        self.layouts['iti_e_x_xx'] = [STATE_ITI, 0,0,0, 0,1,0, 0,1,0, 0,1,0, 1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 2,0, 0,0, 0,1, 0,0,0,0]
        self.layouts['iti_e_x_ne'] = [STATE_ITI, 0,0,1, 0,1,0, 0,1,0, 1,1,0, 1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 3,0,0,0]
        self.layouts['iti_e_x_se'] = [STATE_ITI, 1,0,0, 0,1,1, 0,1,0, 0,1,0, 1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 3,0,0,0]
        self.layouts['iti_s_x_xx'] = [STATE_ITI, 0,1,0, 0,0,0, 0,1,0, 0,1,0, 0,1,0,0, 0,0,0,0, 0,0,0,0, 0,1, 0,0, 2,0, 0,0, 0,0,0,0]
        self.layouts['iti_s_x_se'] = [STATE_ITI, 1,1,0, 0,0,1, 0,1,0, 0,1,0, 0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,3,0,0]
        self.layouts['iti_s_x_sw'] = [STATE_ITI, 0,1,0, 1,0,0, 0,1,1, 0,1,0, 0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,3,0,0]
        self.layouts['iti_w_x_xx'] = [STATE_ITI, 0,1,0, 0,1,0, 0,0,0, 0,1,0, 0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,1, 0,0, 2,0, 0,0,0,0]
        self.layouts['iti_w_x_sw'] = [STATE_ITI, 0,1,0, 1,1,0, 0,0,1, 0,1,0, 0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,3,0]
        self.layouts['iti_w_x_nw'] = [STATE_ITI, 0,1,0, 0,1,0, 1,0,0, 0,1,1, 0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,3,0]
        self.layouts['iti_n_x_xx'] = [STATE_ITI, 0,1,0, 0,1,0, 0,1,0, 0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,0, 2,0, 0,0, 0,1, 0,0, 0,0,0,0]
        self.layouts['iti_n_x_nw'] = [STATE_ITI, 0,1,0, 0,1,0, 1,1,0, 0,0,1, 0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,3]
        self.layouts['iti_n_x_ne'] = [STATE_ITI, 0,1,1, 0,1,0, 0,1,0, 1,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,3]

        # Pretrial configurations: agent confined in start arm before trial begins
        # Naming: pre_{start_arm}_{cue}_xx (xx = no goal wells active)
        # state_type = 3 for all pretrials; arm encoded via value 4 at indexes 33-36
        # Index 33→east, 34→south, 35→west, 36→north (matches ITI trigger position convention)
        # Same barriers as trial base layout + confinement barrier at arm-center junction
        # Confinement barriers: n=(6,5) idx16, e=(7,6) idx13, s=(6,7) idx14, w=(5,6) idx15
        base_pre_layouts = {
            'n' : [STATE_PRETRIAL, 0,0,0, 1,0,1, 0,0,0, 1,0,1, 0,1,0,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,0,4],
            'e' : [STATE_PRETRIAL, 1,0,1, 0,0,0, 1,0,1, 0,0,0, 1,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 4,0,0,0],
            's' : [STATE_PRETRIAL, 0,0,0, 1,0,1, 0,0,0, 1,0,1, 0,1,0,1, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,4,0,0],
            'w' : [STATE_PRETRIAL, 1,0,1, 0,0,0, 1,0,1, 0,0,0, 1,0,1,0, 0,0,0,0, 0,0,0,0, 0,0, 0,0, 0,0, 0,0, 0,0,4,0],
        }
        for start_arm in start_arms:
            for cue in cues:
                variable_name = f'pre_{start_arm}_{cue}_xx'
                layout = base_pre_layouts[start_arm].copy()
                layout[17] = 1 if cue == 'e' else 0   # CUE_LOCATIONS[0]=(11,6) = east
                layout[18] = 1 if cue == 's' else 0   # CUE_LOCATIONS[1]=(6,11) = south
                layout[19] = 1 if cue == 'w' else 0   # CUE_LOCATIONS[2]=(1,6)  = west
                layout[20] = 1 if cue == 'n' else 0   # CUE_LOCATIONS[3]=(6,1)  = north
                self.layouts[variable_name] = layout

        # convert layout values to tuples
        self.layouts.update({k: tuple(v) for k, v in self.layouts.items()})
        # reverse lookup for layout names: used to get embedding object based on state
        self.layout_name_lookup = {v: k for k, v in self.layouts.items()}

        # Per-obs-mode lookup tables (None = use live render fallback).
        self._pose_to_view = None
        self._zero_view = np.zeros((AGENT_VIEW_SIZE, AGENT_VIEW_SIZE, 3), dtype=np.uint8)

        # Load embedding data when needed for obs or human render
        if self.obs_mode in ("embedding", "stereo") or getattr(self, 'render_mode', None) == "human":
            self._load_embeddings()

        # Load precomputed views for fast obs_mode='view'
        if self.obs_mode == "view":
            self._load_views()

        # Exposure A lookup: frozenset of active well indices → layout tuple
        self.maze_config_expa_lookup = {}
        for r in range(1, 5):
            for combo in combinations(range(4), r):
                name = 'expa_x_x_' + '_'.join(well_names[i] for i in combo)
                self.maze_config_expa_lookup[frozenset(combo)] = self.layouts[name]

        # configure lists for pulling layouts during session sequence construction
        # Initialize a 3D list to store the layouts for index accessing with goals, cues, and start_arms
        self.maze_config_trl_list = [[[None for _ in goals] for _ in cues] for _ in start_arms]

        # Create 3d Matrix of maze trial configurations for building session layout order
        for i, startarm in enumerate(start_arms):
            for j, cue in enumerate(cues):
                for k, goal in enumerate(goals):
                    layout_name = f'trl_{startarm}_{cue}_{goal}'
                    self.maze_config_trl_list[i][j][k] = self.layouts.get(layout_name)  # Fetch the layout or None if not found

        # Create 2D Matrix of iti configurations for building session layout order
        self.maze_config_iti_list = [[None for _ in range(3)] for _ in range(4)]
        self.maze_config_iti_list[0][0] = self.layouts.get('iti_n_x_xx')
        self.maze_config_iti_list[0][1] = self.layouts.get('iti_n_x_nw')
        self.maze_config_iti_list[0][2] = self.layouts.get('iti_n_x_ne')
        self.maze_config_iti_list[1][0] = self.layouts.get('iti_e_x_xx')
        self.maze_config_iti_list[1][1] = self.layouts.get('iti_e_x_ne')
        self.maze_config_iti_list[1][2] = self.layouts.get('iti_e_x_se')
        self.maze_config_iti_list[2][0] = self.layouts.get('iti_s_x_xx')
        self.maze_config_iti_list[2][1] = self.layouts.get('iti_s_x_se')
        self.maze_config_iti_list[2][2] = self.layouts.get('iti_s_x_sw')
        self.maze_config_iti_list[3][0] = self.layouts.get('iti_w_x_xx')
        self.maze_config_iti_list[3][1] = self.layouts.get('iti_w_x_sw')
        self.maze_config_iti_list[3][2] = self.layouts.get('iti_w_x_nw')

        # Create 2D Matrix of pretrial configurations: [start_arm][cue]
        self.maze_config_pre_list = [[None for _ in cues] for _ in start_arms]
        for i, startarm in enumerate(start_arms):
            for j, cue in enumerate(cues):
                layout_name = f'pre_{startarm}_{cue}_xx'
                self.maze_config_pre_list[i][j] = self.layouts.get(layout_name)

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
        # 0=left, 1=right, 2=forward, 3=pickup (well entry), 4=pause (no-op)
        self.action_space = spaces.Discrete(5)
        
        # action space and masking control/conditional variables
        self.last_pose  = (None, None, None)
        self.trajectory = []

        # scoring and session trial variables
        self.turn_score = [None, None]
        self.trial_score = None
        self.session_reward = 0
        self.session_num_trials = None
        self.trial_tags = []
        self.trial_configs = []
        self.sequence_labels = []
        self.trial_count = None
        self.episode_trial_scores = []
        self.episode_turn_scores = []
        self.episode_scores = []
        self.phase_step_count = None
        self.session_phase = None  # STATE_EXPA/EXPB/PRETRIAL/TRIAL/ITI
        self.iti_type = None # 0: proximal, 1: distal (only meaningful when session_phase == STATE_ITI)
        self.pretrial_step_count = 0

        # Exposure session variables (shared by expa and expb Phase B)
        self.exposure_wells_remaining = None
        self.exposure_iti_counter = None
        self.exposure_iti_threshold = None
        self.exposure_reward_count = None

        # Exposure B Phase A variables (progressive barrier lowering)
        self.expb_barrier_step = None
        self.expb_delay_counter = 0
        self.expb_delay_target = 0
        self.expb_waiting_for_zone = False
        self.expb_target_zone = None

        # Episode data collection (list-of-dicts, built into DataFrame on demand)
        self.episode = 0
        self.episode_data_rows = []

        # Working-memory reward timers: one per well corner, tracks steps since
        # last reward at that well.  Initialised to n_wm_units (= "no recent
        # reward").  Incremented every step in step(); reset to 0 on well reward.
        # Used by custom_rl.generate_state_vector* for the WM component.
        self.n_wm_units = 10
        self.reward_timers = [self.n_wm_units] * 4

        # Position variables
        self.fwd_pos = None
        self.fwd_cell = None
        self.cur_cell = None

        # Temp single trial session variables
        self.pseudo_session_score = deque([0] * ACQUISITION_SESSION_TRIALS, maxlen=ACQUISITION_SESSION_TRIALS)

        # Training criterion state
        self._criterion_met = False
        self._consecutive_passing_count = 0

        # Policy mode label for data recording ('frozen' or 'updating')
        self.policy_mode = 'updating'

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

    # --- Embedding observation methods ---

    def _load_embeddings(self):
        """Load embedding data and build pose-to-vector lookup dicts."""
        import os
        from pathlib import Path
        # Search several candidate locations so this works for:
        #   (1) editable installs from a local clone (parents[3] = repo root)
        #   (2) Colab / pip-installed wheels with parquet pre-staged in CWD
        #   (3) explicit absolute path supplied via EMBEDDING_PARQUET_PATH
        repo_root = Path(__file__).resolve().parents[3]
        candidates = [
            repo_root / EMBEDDING_PARQUET_PATH,
            Path.cwd() / EMBEDDING_PARQUET_PATH,
            Path(EMBEDDING_PARQUET_PATH),  # in case it's already absolute
        ]
        parquet_path = next((p for p in candidates if p.is_file()), None)
        if parquet_path is None:
            tried = "\n  ".join(str(p) for p in candidates)
            raise FileNotFoundError(
                f"Embeddings parquet not found. Searched:\n  {tried}"
            )
        emb_df = pd.read_parquet(parquet_path)

        self._pose_to_embedding = {}
        self._pose_to_left_eye = {}
        self._pose_to_right_eye = {}

        for _, row in emb_df.iterrows():
            emb_vec = np.array(row['embedding'], dtype=np.float64)
            left_img = np.stack(row['left_eye_img']).astype(np.float64)
            right_img = np.stack(row['right_eye_img']).astype(np.float64)
            for pose_label in row['poses']:
                self._pose_to_embedding[pose_label] = emb_vec
                self._pose_to_left_eye[pose_label] = left_img
                self._pose_to_right_eye[pose_label] = right_img

        self._zero_embedding = np.zeros(EMBEDDING_DIM, dtype=np.float64)
        self._zero_eye = np.zeros((EYE_IMG_SIZE, EYE_IMG_SIZE), dtype=np.float64)

        # Load 96x96 stereo images from dataset bundle for stereo obs mode
        if self.obs_mode == "stereo":
            self._pose_to_stereo = {}
            dataset_path = os.path.join(
                os.path.dirname(__file__), '..', '..', STEREO_DATASET_PATH
            )
            with np.load(os.path.join(dataset_path, 'x.npz'), allow_pickle=False) as npz:
                x_data = npz['data']  # (295, 2, 96, 96) float32
            import json
            with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
                meta = json.load(f)
            label_names = meta['label_names']

            # Build label_name → stereo pair mapping from dataset
            name_to_stereo = {}
            for i, name in enumerate(label_names):
                # x_data[i] is (2, 96, 96) CHW; transpose to (96, 96, 2) HWC
                stereo = np.transpose(x_data[i], (1, 2, 0)).astype(np.float32)
                name_to_stereo[name] = stereo

            # Map poses → stereo using the parquet's label_name → poses mapping
            for _, row in emb_df.iterrows():
                label_name = row['label_name']
                stereo = name_to_stereo.get(label_name)
                if stereo is not None:
                    for pose_label in row['poses']:
                        self._pose_to_stereo[pose_label] = stereo

            self._zero_stereo = np.zeros(
                (STEREO_IMG_SIZE, STEREO_IMG_SIZE, 2), dtype=np.float32
            )

    def _load_views(self):
        """Load the precomputed egocentric-view lookup for obs_mode='view'.

        Avoids re-running get_pov_render_mod on every step. The parquet is
        generated by scripts/build_minigrid_view_lookup.py and is bit-exact
        with the live render across all session types. If the file isn't
        present, self._pose_to_view stays None and _build_observation falls
        back to the live render path.
        """
        import warnings
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        candidates = [
            repo_root / VIEW_PARQUET_PATH,
            Path.cwd() / VIEW_PARQUET_PATH,
            Path(VIEW_PARQUET_PATH),
        ]
        parquet_path = next((p for p in candidates if p.is_file()), None)
        if parquet_path is None:
            warnings.warn(
                f"View lookup parquet not found at any of {[str(p) for p in candidates]}; "
                "falling back to live get_pov_render_mod on every step.",
                stacklevel=2,
            )
            return  # leave self._pose_to_view as None

        vdf = pd.read_parquet(parquet_path)
        self._pose_to_view = {
            r.pose_label: np.frombuffer(r.view, dtype=np.uint8).reshape(
                AGENT_VIEW_SIZE, AGENT_VIEW_SIZE, 3
            )
            for r in vdf.itertuples()
        }

    def _get_pose_label(self) -> str:
        """Construct the embedding pose label from current environment state.

        Looks up the original layout tuple from grid_configuration_sequence
        (which preserves trigger values) rather than maze_state_array (which
        may differ after update_grid_configuration modifies trigger cells).
        """
        x, y = self.agent_pos
        d = self.agent_dir

        # Get the original config tuple from the sequence
        seq_entry = self.grid_configuration_sequence[self.sequence_count]
        if isinstance(seq_entry, tuple):
            layout_name = self.layout_name_lookup.get(seq_entry, '')
        elif isinstance(seq_entry, list):
            # ITI entries are lists of sub-configs; use the selected one
            idx = getattr(self, '_iti_config_idx', 0)
            layout_name = self.layout_name_lookup.get(seq_entry[idx], '') if seq_entry else ''
        else:
            layout_name = ''

        # Expb sessions transition through stages by mutating the grid in-place
        # via EXPB_BARRIER_SEQUENCE, and at the end transition to a STATE_EXPA
        # all-wells layout — both paths bypass the sequence_count machinery, so
        # the static seq_entry above is wrong throughout the session. Reconstruct
        # the active layout from maze_state_array (always up-to-date with the
        # live grid). maze_state_array[0] isn't touched by update_grid_configuration,
        # so we try both STATE_EXPB and STATE_EXPA prefixes before giving up.
        if getattr(self, 'session_type', None) == 'exposure_b':
            tail = tuple(self.maze_state_array[1:])
            for state_type in (STATE_EXPB, STATE_EXPA):
                live_name = self.layout_name_lookup.get((state_type,) + tail)
                if live_name is not None:
                    layout_name = live_name
                    break

        if not layout_name:
            return f"x_x_x_xx_{x}_{y}_{d}"

        parts = layout_name.split('_')
        phase = parts[0]

        if phase == 'expa':
            # Embeddings use 'expa_x_x_xx' regardless of well state
            prefix = 'expa_x_x_xx'
        elif phase in ('trl', 'pre'):
            # trl_{arm}_{cue}_{goal} → trl_{arm}_{cue}_xx (drop goal)
            # pre_{arm}_{cue}_xx → use as-is
            arm = parts[1]
            cue = parts[2]
            prefix = f'{phase}_{arm}_{cue}_xx'
        else:
            # iti and expb: use layout name as-is
            prefix = layout_name

        return f'{prefix}_{x}_{y}_{d}'

    def get_pose_label(self) -> str:
        """Public accessor for the current pose label (layout_class_x_y_dir)."""
        return self._get_pose_label()

    def _build_observation(self) -> dict:
        """Build the observation dict based on the current obs_mode."""
        if self.obs_mode == "view":
            if self._pose_to_view is not None:
                # .copy() preserves the contract that obs are fresh arrays each
                # step — without it, downstream code that writes into the obs
                # (SB3 wrappers, callbacks) would corrupt the cache for every
                # future step that returns the same view.
                img = self._pose_to_view.get(self._get_pose_label(), self._zero_view).copy()
            else:
                img = self.get_pov_render_mod(tile_size=VIEW_TILE_SIZE)
            return {'image': img, 'direction': self.agent_dir, 'mission': self.mission}

        pose_label = self._get_pose_label()
        if self.obs_mode == "stereo":
            stereo = self._pose_to_stereo.get(pose_label, self._zero_stereo)
            return {'stereo': stereo, 'direction': self.agent_dir, 'mission': self.mission}

        emb = self._pose_to_embedding.get(pose_label, self._zero_embedding)
        return {'embedding': emb, 'direction': self.agent_dir, 'mission': self.mission}

    # Grid building functions
    def put_obj_rect(self, obj, topX, topY, width, height):
        for x in range(topX, topX + width):
            for y in range(topY, topY + height):
                self.grid.set(x, y, obj)

    def update_grid_configuration(self, grid_configuration: tuple[int, ...]) -> None:
        """Apply a layout configuration to the grid.

        Updates barriers, cues, wells, and triggers to match the given
        configuration array. Only modifies cells that differ from the
        current maze_state_array.

        Args:
            grid_configuration: 37-element tuple encoding the layout state.
                Index 0: state_type, 1-16: barriers, 17-20: cues,
                21-24: wells, 25-36: triggers.
        """
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
                    self.put_obj(Well(has_reward=True), wl[0], wl[1])
                    self.maze_state_array[i+21] = 1
                else:
                    self.put_obj(Well(), wl[0], wl[1])
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
        """Reset the environment for a new episode.

        Reinitializes scores, trajectories, grid layout, and agent position.
        Returns the initial observation and info dict with action mask.
        """
        super().reset(seed=seed)

        # reset turn score and trial score
        self.turn_score = [None, None]
        self.trial_score = None
        self.session_phase = None
        self.iti_type = None
        self.episode_trial_scores = []
        self.episode_turn_scores = []
        self.session_num_trials = None
        self.trial_tags = []
        self.trial_configs = []
        self.sequence_labels = []
        self.trial_count = 0 # tracks current trial
        self.sequence_count = 0 # used to track sequence position
        self.phase_step_count = 0
        self.pretrial_step_count = 0
        self.session_reward = 0

        # Exposure session reset (shared by expa and expb Phase B)
        self.exposure_wells_remaining = set(range(4))
        self.exposure_iti_threshold = int(random.gauss(EXPOSURE_ITI_STEPS, EXPOSURE_ITI_STD))
        self.exposure_iti_counter = self.exposure_iti_threshold  # ready for first reward immediately
        self.exposure_reward_count = 0

        # Exposure B Phase A reset
        if self.session_type == 'exposure_b':
            self.expb_barrier_step = 0
            self.expb_delay_counter = 0
            self.expb_delay_target = EXPB_ACCLIMATION_STEPS
            self.expb_waiting_for_zone = False
            self.expb_target_zone = None
        else:
            self.expb_barrier_step = None

        # Reset working-memory reward timers (no recent rewards)
        self.reward_timers = [self.n_wm_units] * 4

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

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs_mod = self._build_observation()
        info = {}
        return obs_mod, info

    def gen_grid_configuration_sequence(self) -> tuple[list, int]:
        """Generate the full sequence of grid configurations for the session.

        Builds trial sequences based on session_type and orientation, shuffles
        them with validation constraints, and constructs the interleaved
        pretrial/trial/ITI configuration sequence.

        Returns:
            Tuple of (grid_configuration_sequence, num_trials).
        """
        goal_location_index = GOAL_LOCATION_MAP.get(
            self.start_goal_location, random.randint(0, NUM_ARMS - 1)
        )

        ori = self.agent_cue_goal_orientation
        gli = goal_location_index

        # --- Helpers: generate shuffled + validated trial lists ---

        def _gen_simple_pool(pool, chunk_size, threepeat_limit):
            """Shuffle uniform chunks from pre-built pool, validate start-only repeats."""
            while True:
                result = shuffle_uniform_chunks(pool, chunk_size)
                if validate_sequence_start_only(result, threepeat_limit):
                    return [(s, c, g, tag) for s, c, r, g, tag in result]

        def _gen_multi_pool(pool, chunk_size, start_limit=3, route_limit=3, goal_limit=3):
            """Shuffle uniform chunks, validate multi-index repeats."""
            while True:
                result = shuffle_uniform_chunks(pool, chunk_size)
                if validate_sequence_multi(result, start_limit, route_limit, goal_limit):
                    return [(s, c, g, tag) for s, c, r, g, tag in result]

        def _gen_novel_route(trained_pairs, probe_pairs, chunk_size, threepeat_limit=0):
            """Acq-then-novel shuffle with start-only validation."""
            trained = list(trained_pairs)
            probe = list(probe_pairs)
            while True:
                result, success = shuffle_acq_then_novel(trained, probe, chunk_size)
                if not success:
                    continue
                if validate_sequence_start_only(result, threepeat_limit):
                    return [(s, c, g, tag) for s, c, r, g, tag in result]

        def _gen_novel_route_multi(trained_pairs, probe_pairs, chunk_size,
                                   start_limit=3, route_limit=3, goal_limit=3):
            """Acq-then-novel shuffle with multi-index validation."""
            trained = list(trained_pairs)
            probe = list(probe_pairs)
            while True:
                result, success = shuffle_acq_then_novel(trained, probe, chunk_size)
                if not success:
                    continue
                if validate_sequence_multi(result, start_limit, route_limit, goal_limit):
                    return [(s, c, g, tag) for s, c, r, g, tag in result]

        def _gen_reversal(trained_pairs, probe_pairs, chunk_size, threepeat_limit=0):
            """Acq-then-probe shuffle with start-only validation."""
            trained = list(trained_pairs)
            probe = list(probe_pairs)
            while True:
                result = shuffle_acq_then_probe(trained, probe, chunk_size)
                if validate_sequence_start_only(result, threepeat_limit):
                    return [(s, c, g, tag) for s, c, r, g, tag in result]

        def _gen_reversal_multi(trained_pairs, probe_pairs, chunk_size,
                                start_limit=3, route_limit=3, goal_limit=3):
            """Acq-then-probe shuffle with multi-index validation."""
            trained = list(trained_pairs)
            probe = list(probe_pairs)
            while True:
                result = shuffle_acq_then_probe(trained, probe, chunk_size)
                if validate_sequence_multi(result, start_limit, route_limit, goal_limit):
                    return [(s, c, g, tag) for s, c, r, g, tag in result]

        # --- Generator functions for each session type ---

        def gen_pi_vc_f2_single_trial():
            return [(random.choice([1, 3]), 0, gli, 'trained')]

        def gen_pi_vc_f2_acq():
            pool = get_f2_trained_pairs(ori, gli) * 4
            return _gen_simple_pool(pool, chunk_size=4, threepeat_limit=2)

        def gen_pi_vc_f2_novel_route():
            trained = get_f2_trained_pairs(ori, gli) * 4
            probe = get_f2_novel_route_probe_pairs(ori, gli)
            return _gen_novel_route(trained, probe, chunk_size=6)

        def gen_pi_vc_f2_no_cue():
            pool = get_f2_no_cue_pairs(ori, gli) * 4
            return _gen_simple_pool(pool, chunk_size=4, threepeat_limit=2)

        def gen_pi_vc_f2_rotate():
            pool = get_f2_rotate_pairs(ori)
            return _gen_multi_pool(pool, chunk_size=2)

        def gen_pi_vc_f2_reversal():
            trained = get_f2_trained_pairs(ori, gli) * 4
            probe = get_f2_reversal_probe_pairs(ori, gli) * 4
            return _gen_reversal(trained, probe, chunk_size=10)

        def gen_pi_vc_f1_acq():
            pool = get_f1_trained_pairs(ori, gli) * 4
            return _gen_simple_pool(pool, chunk_size=4, threepeat_limit=2)

        def gen_pi_vc_f1_novel_route():
            trained = get_f1_trained_pairs(ori, gli) * 4
            probe = get_f1_novel_route_probe_pairs(ori, gli)
            return _gen_novel_route(trained, probe, chunk_size=6)

        def gen_pi_vc_f1_no_cue():
            pool = get_f1_no_cue_pairs(ori, gli) * 4
            return _gen_simple_pool(pool, chunk_size=4, threepeat_limit=2)

        def gen_pi_vc_f1_rotate():
            pool = get_f1_rotate_pairs(ori)
            return _gen_multi_pool(pool, chunk_size=2)

        def gen_pi_vc_f1_reversal():
            trained = get_f1_trained_pairs(ori, gli) * 4
            probe = get_f1_reversal_probe_pairs(ori, gli) * 4
            return _gen_reversal(trained, probe, chunk_size=10)

        def gen_pi_novel_route_no_cue():
            trained = get_f2_no_cue_pairs(ori, gli) * 4
            probe = get_pi_novel_route_no_cue_probe_pairs(ori, gli)
            return _gen_novel_route(trained, probe, chunk_size=6)

        def gen_pi_reversal_no_cue():
            trained = get_f2_no_cue_pairs(ori, gli) * 4
            probe = get_pi_reversal_no_cue_probe_pairs(ori, gli) * 4
            return _gen_reversal(trained, probe, chunk_size=10)

        def gen_vc_acquisition():
            pool = get_f2_rotate_pairs(ori)
            return _gen_multi_pool(pool, chunk_size=4)

        def gen_vc_novel_route_rotate():
            trained = get_f2_rotate_pairs(ori)
            probe = get_vc_novel_route_rotate_probe_pairs(ori)
            return _gen_novel_route_multi(trained, probe, chunk_size=3)

        def gen_vc_reversal_rotate():
            trained = get_f2_rotate_pairs(ori)
            probe = get_vc_reversal_rotate_probe_pairs(ori)
            return _gen_reversal_multi(trained, probe, chunk_size=10)

        def gen_exposure():
            return []  # exposure doesn't use start_goal_cue_list

        def gen_exposure_b():
            return []  # exposure_b doesn't use start_goal_cue_list

        # --- Dispatch table ---
        SESSION_GENERATORS = {
            'exposure': gen_exposure,
            'exposure_b': gen_exposure_b,
            'PI+VC f2 single trial': gen_pi_vc_f2_single_trial,
            'PI+VC f2 acquisition': gen_pi_vc_f2_acq,
            'PI+VC f2 novel route': gen_pi_vc_f2_novel_route,
            'PI+VC f2 no cue': gen_pi_vc_f2_no_cue,
            'PI+VC f2 rotate': gen_pi_vc_f2_rotate,
            'PI+VC f2 reversal': gen_pi_vc_f2_reversal,
            'PI+VC f1 acquisition': gen_pi_vc_f1_acq,
            'PI+VC f1 novel route': gen_pi_vc_f1_novel_route,
            'PI+VC f1 no cue': gen_pi_vc_f1_no_cue,
            'PI+VC f1 rotate': gen_pi_vc_f1_rotate,
            'PI+VC f1 reversal': gen_pi_vc_f1_reversal,
            'PI acquisition': gen_pi_vc_f2_no_cue,        # same session design
            'PI novel route no cue': gen_pi_novel_route_no_cue,
            'PI novel route cue': gen_pi_vc_f2_novel_route,
            'PI reversal no cue': gen_pi_reversal_no_cue,
            'PI reversal cue': gen_pi_vc_f2_reversal,
            'VC acquisition': gen_vc_acquisition,
            'VC novel route fixed': gen_pi_vc_f2_novel_route,
            'VC novel route rotate': gen_vc_novel_route_rotate,
            'VC reversal fixed': gen_pi_vc_f2_reversal,
            'VC reversal rotate': gen_vc_reversal_rotate,
        }

        generator = SESSION_GENERATORS.get(self.session_type)
        if generator is None:
            print('Invalid session type')
            start_goal_cue_list = None
        elif self._fixed_trial_configs:
            # Yoked replay path: skip the random SESSION_GENERATORS shuffle
            # and use the rat's actual (start_arm, cue, goal, tag) trials
            # so trigger cells fire at the same step the rat experienced
            # them. The dataset stores trial_configs as a list of
            # [arm, cue, goal, tag] entries — same 4-shape the rest of
            # this method consumes.
            start_goal_cue_list = [tuple(tc) for tc in self._fixed_trial_configs]
        else:
            start_goal_cue_list = generator()

        # --- Exposure sessions: bypass normal sequence building ---
        if self.session_type == 'exposure':
            all_wells = self.maze_config_expa_lookup[frozenset(range(4))]
            self.trial_configs = []
            self.sequence_labels = []
            return [all_wells], EXPOSURE_NUM_REWARDS, []

        if self.session_type == 'exposure_b':
            initial_layout = self.layouts['expb_xxxx_x_xx']
            self.trial_configs = []
            self.sequence_labels = []
            return [initial_layout], EXPB_NUM_REWARDS, []

        # --- Extract trial tags and configs from the tagged tuples ---
        trial_tags = [sgc[3] for sgc in start_goal_cue_list] if start_goal_cue_list else []
        self.trial_configs = [list(sgc) for sgc in start_goal_cue_list] if start_goal_cue_list else []

        # --- Build grid configuration sequence ---
        grid_configuration_sequence = []
        len_sgc = len(start_goal_cue_list)
        num_trials = len_sgc
        for i, sgc in enumerate(start_goal_cue_list):
            start_arm = sgc[0]
            cue = sgc[1]
            goal = sgc[2]
            if len_sgc == 1:
                grid_configuration_sequence.append(self.maze_config_pre_list[start_arm][cue])
                grid_configuration_sequence.append(self.maze_config_trl_list[start_arm][cue][goal])
            elif len_sgc > 1:
                if i < len_sgc - 1:
                    next_start_arm = start_goal_cue_list[i + 1][0]
                grid_configuration_sequence.append(self.maze_config_pre_list[start_arm][cue])
                grid_configuration_sequence.append(self.maze_config_trl_list[start_arm][cue][goal])
                if i < len_sgc - 1:
                    grid_configuration_sequence.append(self.maze_config_iti_list[next_start_arm])
            else:
                grid_configuration_sequence = None

        # --- Build sequence labels from layout name lookup ---
        sequence_labels = []
        for entry in grid_configuration_sequence:
            if isinstance(entry, tuple):
                sequence_labels.append(self.layout_name_lookup.get(entry, ''))
            elif isinstance(entry, list):
                # ITI entries are [distal, proximal_adj, proximal] — store all 3 names
                sequence_labels.append([self.layout_name_lookup.get(sub, '') for sub in entry])
        self.sequence_labels = sequence_labels

        return grid_configuration_sequence, num_trials, trial_tags

    def gen_start_pose(self) -> tuple[tuple[int, int], int]:
        """Determine the agent's starting position and direction for the current trial.

        Parses the layout name from the first grid configuration to identify
        the start arm, then returns the corresponding corner position and
        facing direction.
        """
        first_config = self.grid_configuration_sequence[0]
        layout_name = self.layout_name_lookup.get(first_config, '')
        parts = layout_name.split('_')
        if len(parts) >= 2:
            # Exposure layouts start at center
            if parts[0] == 'expa':
                return (6, 6), DEFAULT_AGENT_START_DIR
            if parts[0] == 'expb':
                return (6, 6), random.randint(0, 3)
            arm = parts[1]
            arm_to_pose = {
                'n': ((6, 3), 1),  # North start arm
                'e': ((9, 6), 2),  # East start arm
                's': ((6, 9), 3),  # South start arm
                'w': ((3, 6), 0),  # West start arm
            }
            if arm in arm_to_pose:
                return arm_to_pose[arm]
        # Fallback to barrier-index detection for non-standard layouts
        if self.grid_configuration_sequence[0][13] == 1:
            return (3, 6), 0  # West start arm
        elif self.grid_configuration_sequence[0][14] == 1:
            return (6, 3), 1  # North start arm
        elif self.grid_configuration_sequence[0][15] == 1:
            return (9, 6), 2  # East start arm
        elif self.grid_configuration_sequence[0][16] == 1:
            return (6, 9), 3  # South start arm
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

    def render(self):
        """Render the environment, adding eye images when embeddings are loaded."""
        if self.render_mode != "human" or not hasattr(self, '_pose_to_left_eye') or not hasattr(self, 'sequence_count'):
            return super().render()

        import pygame
        import pygame.freetype

        # Get grid image
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)
        img = np.transpose(img, axes=(1, 0, 2))

        if self.render_size is None:
            self.render_size = img.shape[:2]

        # Look up eye images for current pose
        pose_label = self._get_pose_label()
        left_eye = self._pose_to_left_eye.get(pose_label, self._zero_eye)
        right_eye = self._pose_to_right_eye.get(pose_label, self._zero_eye)

        # Convert float64 [0,1] to uint8 [0,255] RGB for pygame
        left_rgb = np.repeat((left_eye * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2)
        right_rgb = np.repeat((right_eye * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2)

        # Create surfaces
        grid_surf = pygame.surfarray.make_surface(img)
        grid_size = grid_surf.get_size()

        # Scale eye images to match grid height
        eye_display_h = grid_size[1]
        eye_display_w = eye_display_h  # keep square

        left_surf = pygame.surfarray.make_surface(np.transpose(left_rgb, (1, 0, 2)))
        left_surf = pygame.transform.smoothscale(left_surf, (eye_display_w, eye_display_h))
        right_surf = pygame.surfarray.make_surface(np.transpose(right_rgb, (1, 0, 2)))
        right_surf = pygame.transform.smoothscale(right_surf, (eye_display_w, eye_display_h))

        # Layout: [grid] [gap] [left_eye] [gap] [right_eye]
        #         [session]     [pose_label         ]
        gap = int(grid_size[0] * 0.05)
        total_w = grid_size[0] + gap + eye_display_w + gap + eye_display_w
        font_size = 18
        text_row_h = int(font_size * 1.8)
        padding = int(grid_size[0] * 0.03)

        window_w = total_w + padding * 2
        window_h = grid_size[1] + text_row_h + padding * 2
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((window_w, window_h))
            pygame.display.set_caption("Corner Maze")
        elif self.window.get_size() != (window_w, window_h):
            self.window = pygame.display.set_mode((window_w, window_h))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        # Blit grid and eye images
        self.window.blit(grid_surf, (padding, padding))
        eye_x_left = padding + grid_size[0] + gap
        eye_x_right = eye_x_left + eye_display_w + gap
        self.window.blit(left_surf, (eye_x_left, padding))
        self.window.blit(right_surf, (eye_x_right, padding))

        # Text row below images
        font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
        text_y = padding + grid_size[1] + 4

        # Session info under grid
        session_text = f'{self.session_type}'
        font.render_to(self.window, (padding, text_y), session_text, size=font_size, fgcolor=(0, 0, 0))

        # Pose label centered under the eye images
        label_center_x = eye_x_left + (eye_display_w + gap + eye_display_w) // 2
        label_rect = font.get_rect(pose_label, size=font_size)
        font.render_to(self.window, (label_center_x - label_rect.width // 2, text_y), pose_label, size=font_size, fgcolor=(0, 0, 0))

        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def plot_observation(self, observation_rgb):
        """
        Plots the full RGB observation along with individual R, G, B channels using matplotlib,
        without creating a new window each time.
        """
        import matplotlib.pyplot as plt

        # Check if the figure already exists
        if not hasattr(self, 'fig'):
            self.fig, self.axes = plt.subplots(1, 4, figsize=(16, 4))

        for ax in self.axes:
            ax.clear()

        self.axes[0].imshow(observation_rgb, interpolation='nearest')
        self.axes[0].set_title("Full RGB")
        self.axes[0].axis('off')

        channel_titles = ["Cues", "Walls", "Ledges"]
        for i in range(3):
            channel_img = observation_rgb[:, :, i]
            self.axes[i + 1].imshow(channel_img, cmap="gray", vmin=0, vmax=255, interpolation='nearest')
            self.axes[i + 1].set_title(channel_titles[i])
            self.axes[i + 1].axis('off')

        plt.draw()
        plt.pause(0.001)
    
    # ---- Private helpers for step() ----

    def _apply_action(self, action: int) -> str:
        """Process the given action, update agent position/direction, return event type."""
        if action == Actions.left:
            # Turn left after exiting a well moves forward in one step
            if self.agent_pos in CORNERS and self.last_pose in WELL_EXIT_POSES:
                new_pos_dir = CORNER_LEFT_TURN_WELL_EXIT.get(self.agent_pos)
                if new_pos_dir:
                    self.agent_pos, self.agent_dir = new_pos_dir
                return "forward"
            else:
                self.agent_dir = (self.agent_dir - 1) % 4
                return "turn"
        elif action == Actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
            return "turn"
        elif action == Actions.forward:
            if self.agent_pose in WELL_EXIT_POSES:
                new_pos_dir = WELL_EXIT_FORWARD.get(self.agent_pos)
                if new_pos_dir:
                    self.agent_pos, self.agent_dir = new_pos_dir
                return "forward"
            elif self.fwd_cell is None or self.fwd_cell.can_overlap():
                self.agent_pos = tuple(self.fwd_pos)
                return "forward"
            else:
                return "wall_bump"
        elif action == ACTION_ENTER_WELL:
            new_pos_dir = WELL_ENTRY_PICKUP.get(self.agent_pos)
            if new_pos_dir:
                self.agent_pos, self.agent_dir = new_pos_dir
            return "forward"
        elif action == ACTION_PAUSE:
            return "pause"
        return "forward"

    def _update_state(self, action: int = None) -> None:
        """Update cell type, pose tracking, and trajectory after an action."""
        self.cur_cell = self.grid.get(*self.agent_pos)
        if not (action == ACTION_PAUSE and self.last_pose in WELL_EXIT_POSES):
            self.last_pose = self.agent_pose
        self.agent_pose = (*self.agent_pos, self.agent_dir)
        state_type = self.grid_configuration_sequence[self.sequence_count][0]
        self.trajectory.append((*self.agent_pose, state_type))

    def _update_turn_scores(self) -> None:
        """Record turn-one and turn-two scores based on agent position during trials."""
        config = self.grid_configuration_sequence[self.sequence_count]
        if config[0] != STATE_TRIAL:
            return
        goal_locations = config[21:25]
        ns_goal_active = 1 in (goal_locations[0], goal_locations[2])
        ew_goal_active = 1 in (goal_locations[1], goal_locations[3])
        if self.turn_score[0] is None:
            if ns_goal_active and self.agent_pos in TURN_ONE_SENW_MAP:
                self.turn_score[0] = TURN_ONE_SENW_MAP[self.agent_pos]
            elif ew_goal_active and self.agent_pos in TURN_ONE_NESW_MAP:
                self.turn_score[0] = TURN_ONE_NESW_MAP[self.agent_pos]
        elif self.turn_score[1] is None:
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

    def _log_and_finalize_episode(self) -> None:
        """Single point of truth for episode-end bookkeeping and data capture."""
        self.episode += 1
        episode_score = sum(self.episode_trial_scores) / len(self.episode_trial_scores)
        self.episode_scores.append(episode_score)
        self._evaluate_criterion(episode_score)

        self.episode_data_rows.append({
            'session_type': self.session_type,
            'policy_mode': self.policy_mode,
            'episode': self.episode,
            'trial_scores': list(self.episode_trial_scores),
            'turn_scores': [list(ts) for ts in self.episode_turn_scores],
            'num_trials_completed': self.trial_count,
            'total_steps': self.step_count,
            'total_reward': self.session_reward,
            'trajectory': [list(t) for t in self.trajectory],
            'session_scores': self._compute_session_scores(),
            'trial_tags': list(self.trial_tags),
            'trial_configs': list(self.trial_configs),
            'sequence_labels': list(self.sequence_labels),
        })

    def _compute_session_scores(self) -> dict:
        """Compute phase-specific scores from trial_scores grouped by trial_tags.

        Returns dict with keys like 'trained', 'trained_possible', 'novel',
        'novel_possible', etc. The *_possible values reflect the full designed
        count; the bare keys count correct trials among those completed.
        """
        scores = {}
        for tag in set(self.trial_tags):
            tag_correct = sum(s for s, t in zip(self.episode_trial_scores, self.trial_tags) if t == tag)
            tag_possible = sum(1 for t in self.trial_tags if t == tag)
            scores[tag] = tag_correct
            scores[f'{tag}_possible'] = tag_possible
        return scores

    def _evaluate_criterion(self, episode_score: float) -> None:
        """Update training criterion based on latest episode score.

        Acquisition sessions: criterion met after N consecutive scores >= threshold.
        Non-acquisition sessions (probes, reversals, etc.): criterion met after one session.
        """
        if 'acquisition' not in (self.session_type or ''):
            self._criterion_met = True
            return
        # Only count sessions where all trials completed
        if (self.trial_count == self.session_num_trials
                and episode_score >= self.criterion_threshold):
            self._consecutive_passing_count += 1
        else:
            self._consecutive_passing_count = 0
        if self._consecutive_passing_count >= self.criterion_consecutive:
            self._criterion_met = True

    @property
    def training_criterion_met(self) -> bool:
        return self._criterion_met

    def get_episode_data(self) -> pd.DataFrame:
        """Build DataFrame from collected rows. Call after training."""
        return pd.DataFrame(self.episode_data_rows)

    def get_action_mask(self):
        """Return action validity mask [LEFT, RIGHT, FORWARD, PICKUP, PAUSE].

        PAUSE is always True here; callers override with ENABLE_PAUSE as needed.
        """
        if self.agent_pose in WELL_EXIT_POSES:
            # Still at well position facing inward (before FORWARD exit)
            return [True, False, True, False, True]
        if self.agent_pos in CORNERS and self.last_pose in WELL_EXIT_POSES:
            # At corner, just exited well — only L/F/PAUSE valid
            return [True, False, True, False, True]
        if (*self.agent_pos, self.agent_dir) in CORNER_POSES:
            return [True, True, True, True, True]
        return [True, True, True, False, True]

    def _handle_exposure_well(self) -> tuple[str, bool]:
        """Handle well visit during exposure session (cycle-alternation with step-based ITI)."""
        well_idx = WELL_LOCATIONS.index(tuple(self.agent_pos))
        if well_idx in self.exposure_wells_remaining and self.exposure_iti_counter >= self.exposure_iti_threshold:
            self.reward_timers[well_idx] = 0
            self.exposure_reward_count += 1
            self.exposure_wells_remaining.discard(well_idx)
            self.exposure_iti_counter = 0
            self.exposure_iti_threshold = int(random.gauss(EXPOSURE_ITI_STEPS, EXPOSURE_ITI_STD))
            if len(self.exposure_wells_remaining) == 0:
                self.exposure_wells_remaining = set(range(4))
            next_layout = self.maze_config_expa_lookup[frozenset(self.exposure_wells_remaining)]
            self.update_grid_configuration(next_layout)
            terminated = self.exposure_reward_count >= EXPOSURE_NUM_REWARDS
            return "well_reward", terminated
        else:
            return "well_empty", False

    def _handle_reward_well(self) -> tuple[str, bool]:
        """Handle reward well entry for single-trial and multi-trial sessions."""
        well_idx = WELL_LOCATIONS.index(tuple(self.agent_pos))
        self.reward_timers[well_idx] = 0
        self.trial_count += 1

        if self.session_num_trials == 1:
            if self.trial_score is None:
                self.trial_score = 1
            self.episode_trial_scores.append(self.trial_score)
            self.episode_turn_scores.append(list(self.turn_score))
            self._log_and_finalize_episode()
            self.pseudo_session_score.appendleft(self.trial_score)
            return "well_reward", True

        elif self.session_num_trials > 1:
            if self.trial_score is None:
                self.trial_score = 1
            self.episode_trial_scores.append(self.trial_score)
            self.episode_turn_scores.append(list(self.turn_score))
            # Reset per-trial state
            self.trial_score = None
            self.turn_score = [None, None]
            self.phase_step_count = 0

            if self.trial_count == self.session_num_trials:
                # Last trial — end episode
                self._log_and_finalize_episode()
                return "well_reward", True
            else:
                # Mid-session — transition to ITI
                self._select_iti_configuration()
                return "well_reward", False

        return "well_reward", False

    def _select_iti_configuration(self) -> None:
        """Select and apply the ITI grid configuration based on goal and start arm proximity."""
        self.sequence_count += 1
        # Determine current goal location from well states (indices 21-24)
        prev_config = self.grid_configuration_sequence[self.sequence_count - 1]
        goal_loc = next(
            (i for i in range(4) if prev_config[21 + i] == 1),
            self.grid_configuration_sequence[self.sequence_count][-1],
        )
        # Determine upcoming start arm from barrier openings (indices 2, 5, 8, 11)
        iti_configs = self.grid_configuration_sequence[self.sequence_count]
        start_arm_loc = next(
            (arm for arm, idx in enumerate([2, 5, 8, 11]) if iti_configs[0][idx] == 0),
            self.grid_configuration_sequence[self.sequence_count + 1][-2],
        )
        # Proximal (same/adjacent arm) vs distal
        if start_arm_loc == goal_loc:
            config_idx, self.iti_type = 2, 0
        elif start_arm_loc == (goal_loc + 1) % 4:
            config_idx, self.iti_type = 1, 0
        else:
            config_idx, self.iti_type = 0, 1
        self._iti_config_idx = config_idx
        self.update_grid_configuration(iti_configs[config_idx])
        self.session_phase = STATE_ITI

    def _handle_trigger(self) -> None:
        """Handle A/B/S trigger interactions: update grid config and advance sequence."""
        self.trial_score = None
        self.turn_score = [None, None]
        trigger_type = self.grid.get(*self.agent_pos).get_trigger_type()
        if trigger_type == 'A':
            self._iti_config_idx = 1
            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][1])
        elif trigger_type == 'B':
            self._iti_config_idx = 2
            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count][2])
        elif trigger_type == 'S':
            self.sequence_count += 1
            next_config = self.grid_configuration_sequence[self.sequence_count]
            self.update_grid_configuration(next_config)
            if next_config[0] == STATE_PRETRIAL:
                self.session_phase = STATE_PRETRIAL
                self.pretrial_step_count = 0
            else:
                self.session_phase = STATE_TRIAL
            self.phase_step_count = 0

    def _handle_pretrial(self) -> None:
        """Check pretrial position-based trigger after minimum steps."""
        if self.session_phase != STATE_PRETRIAL:
            return
        self.pretrial_step_count += 1
        if self.pretrial_step_count < PRETRIAL_MIN_STEPS:
            return
        current_config = self.grid_configuration_sequence[self.sequence_count]
        # Map trigger index to arm: 33→east(1), 34→south(2), 35→west(3), 36→north(0)
        arm_idx = None
        for idx, arm in ((33, 1), (34, 2), (35, 3), (36, 0)):
            if current_config[idx] == 4:
                arm_idx = arm
                break
        if arm_idx is not None and tuple(self.agent_pos) == PRETRIAL_TRIGGER_POSITIONS[arm_idx]:
            self.sequence_count += 1
            self.update_grid_configuration(self.grid_configuration_sequence[self.sequence_count])
            self.session_phase = STATE_TRIAL
            self.phase_step_count = 0
            self.pretrial_step_count = 0

    def _handle_exposure_b_phase_a(self) -> None:
        """Progressive barrier lowering state machine for Exposure B Phase A."""
        if self.session_phase != STATE_EXPB or self.expb_barrier_step is None:
            return
        self.expb_delay_counter += 1
        if self.expb_waiting_for_zone:
            if tuple(self.agent_pos) == self.expb_target_zone:
                self.expb_barrier_step += 1
                if self.expb_barrier_step >= len(EXPB_BARRIER_SEQUENCE):
                    self.session_phase = STATE_EXPA
                    self.update_grid_configuration(
                        self.maze_config_expa_lookup[frozenset(range(4))]
                    )
                else:
                    seq = EXPB_BARRIER_SEQUENCE[self.expb_barrier_step]
                    self.update_grid_configuration(self.layouts[seq[0]])
                    self.expb_delay_counter = 0
                    self.expb_delay_target = seq[3]
                    self.expb_waiting_for_zone = False
        elif self.expb_delay_counter >= self.expb_delay_target:
            next_step = self.expb_barrier_step + 1
            if next_step >= len(EXPB_BARRIER_SEQUENCE):
                self.session_phase = STATE_EXPA
                self.update_grid_configuration(
                    self.maze_config_expa_lookup[frozenset(range(4))]
                )
            else:
                seq = EXPB_BARRIER_SEQUENCE[next_step]
                if seq[1] == 'timed':
                    self.expb_barrier_step = next_step
                    self.update_grid_configuration(self.layouts[seq[0]])
                    self.expb_delay_counter = 0
                    self.expb_delay_target = seq[3]
                elif seq[1] == 'zone':
                    self.expb_waiting_for_zone = True
                    self.expb_target_zone = seq[2]

    def _handle_timeout(self) -> None:
        """Handle episode bookkeeping when max_steps is reached."""
        if self.session_num_trials == 1:
            self.trial_score = 0
            self.episode_trial_scores.append(self.trial_score)
            self.episode_turn_scores.append(list(self.turn_score))
            self._log_and_finalize_episode()
            self.pseudo_session_score.appendleft(self.trial_score)
        elif self.session_num_trials > 1:
            if self.trial_score is None:
                self.trial_score = 0
            self.episode_trial_scores.append(self.trial_score)
            self.episode_turn_scores.append(list(self.turn_score))
            self._log_and_finalize_episode()

    # ---- Reward ----

    def _compute_reward(self, well_event: str | None, action: int) -> float:
        """Compute step reward. Single source of truth for all reward values.

        Turns (actions 0, 1) cost STEP_TURN_COST. All other actions cost
        STEP_FORWARD_COST. Reaching a reward well adds WELL_REWARD_SCR.
        """
        reward = STEP_TURN_COST if action in (0, 1) else STEP_FORWARD_COST

        if well_event == "well_reward":
            reward += WELL_REWARD_SCR

        return reward

    # ---- Main step method ----

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one environment step given an action.

        Args:
            action: Action index (0=left, 1=right, 2=forward, 3=pickup, 4=pause).

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        self.step_count += 1
        self.reward_timers = [t + 1 for t in self.reward_timers]
        terminated = False
        truncated = False

        # --- Execute action & update state ---
        self._apply_action(action)
        self._update_state(action)
        self._update_turn_scores()

        # --- Detect well/trigger events ---
        well_event = None
        if isinstance(self.cur_cell, Well) and self.cur_cell.has_reward and self.session_phase == STATE_EXPA:
            well_event, t = self._handle_exposure_well()
            terminated = terminated or t
        elif isinstance(self.cur_cell, Well) and self.cur_cell.has_reward:
            well_event, t = self._handle_reward_well()
            terminated = terminated or t
        elif isinstance(self.cur_cell, Well) and not self.cur_cell.has_reward and self.last_pose[:2] not in WELL_LOCATIONS:
            self.trial_score = 0
        elif self.is_agent_on_obj(Trigger):
            self._handle_trigger()

        # --- Phase logic ---
        self._handle_pretrial()
        self._handle_exposure_b_phase_a()
        if self.session_phase == STATE_EXPA:
            self.exposure_iti_counter += 1
        self.fwd_pos = self.front_pos
        self.fwd_cell = self.grid.get(*self.fwd_pos)
        self.phase_step_count += 1

        # --- Single reward calculation ---
        reward = self._compute_reward(well_event, action)
        self.session_reward += reward

        if self.render_mode == "human":
            self.render()

        # --- Timeout bookkeeping ---
        if self.step_count >= self.max_steps:
            self._handle_timeout()
            truncated = True

        # --- Build observation ---
        obs_mod = self._build_observation()
        info = {'agent_pos': self.agent_pos,
                'terminated': terminated, 'truncated': truncated,
                'episode_scores': self.episode_scores, 'session_reward': self.session_reward,
                'training_criterion_met': self._criterion_met,
                'trial_count': self.trial_count}
        return obs_mod, reward, terminated, truncated, info

    @staticmethod
    def _gen_mission():
        return "corner maze mission"

    def _gen_grid(self, width: int, height: int) -> None:
        """Build the maze grid, apply the initial layout, and place the agent."""
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
        for cx, cy in CUE_LOCATIONS:
            self.put_obj(Wall(color='cue_off_rgb'), cx, cy)
        # Place wells
        for wx, wy in WELL_LOCATIONS:
            self.put_obj(Well(), wx, wy)

        # Build session configuration sequence data
        self.grid_configuration_sequence, self.session_num_trials, self.trial_tags = self.gen_grid_configuration_sequence()

        # Configure maze environment to the first setting of grid_configuration_sequence
        self.update_grid_configuration(self.grid_configuration_sequence[0])

        # Determine start position from grid config and set agent start position and direction
        self.agent_pos, self.agent_dir = self.gen_start_pose()
        self.agent_pose = (*self.agent_pos, self.agent_dir)
        self.agent_start_pos = self.agent_pos
        # Set initial phase based on state_type of first layout
        if self.grid_configuration_sequence[0][0] == STATE_PRETRIAL:
            self.session_phase = STATE_PRETRIAL
            self.pretrial_step_count = 0
        elif self.grid_configuration_sequence[0][0] == STATE_EXPA:
            self.session_phase = STATE_EXPA
        elif self.grid_configuration_sequence[0][0] == STATE_EXPB:
            self.session_phase = STATE_EXPB
        else:
            self.session_phase = STATE_TRIAL

        # Update position
        # Get the position in front of the agent
        self.fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        self.fwd_cell = self.grid.get(*self.fwd_pos)
        # Get the cell type the agent is on
        self.cur_cell = self.grid.get(*self.agent_pos)
        # save agent pose for each step to produce a path history of the session
        state_type = self.grid_configuration_sequence[0][0]
        self.trajectory.append((*self.agent_pose, state_type))

        self.mission = "corner maze mission"

