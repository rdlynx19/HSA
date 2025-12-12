"""
GoalCurriculumManager Module: Dynamic Goal Sampling for Reinforcement Learning.

This module defines the `GoalCurriculumManager` class, which implements an automated 
curriculum learning strategy for goal-based tasks. It dynamically adjusts the difficulty 
of the task by expanding or contracting the maximum distance goals are sampled from, 
based on the agent's recent success rate.

The manager ensures goal positions respect minimum distance constraints (dead zone) 
and tracks performance metrics such as success rate and curriculum progress.
"""
import pickle, os

import numpy as np
from numpy.typing import NDArray

class GoalCurriculumManager:
    """
    Manages the curriculum for goal positions in the environment based on agent performance.

    The curriculum dynamically expands or contracts the maximum sampling distance 
    for goals (radius) based on a rolling average of the agent's recent success rate, 
    ensuring the agent learns progressively harder tasks.
    """
    def __init__(self, 
                 initial_range: tuple[float, float] = (1.5, 2.0),
                 target_range: tuple[float, float] = (1.5, 4.5), 
                 success_threshold: float = 0.75,
                 failure_threshold: float = 0.40,
                 expansion_step: float = 0.3,
                 window_size: int = 100,
                 min_episodes_before_expand: int = 50,
                 dead_zone_radius: float = 1.2,
                 ):
        """
        Initializes the curriculum manager with goal constraints and performance metrics.

        :param initial_range: The starting range (min, max) for goal distances. Expansion begins from the maximum value.
        :type initial_range: tuple[float, float]
        :param target_range: The ultimate range (min, max) the curriculum can reach.
        :type target_range: tuple[float, float]
        :param success_threshold: The success rate threshold (e.g., 0.75) required over the window size to EXPAND the curriculum.
        :type success_threshold: float
        :param failure_threshold: The success rate threshold (e.g., 0.40) below which the curriculum CONTRACTS.
        :type failure_threshold: float
        :param expansion_step: The step size (in meters) by which the curriculum range expands or contracts.
        :type expansion_step: float
        :param window_size: The number of recent episodes to consider for calculating the success/failure rate.
        :type window_size: int
        :param min_episodes_before_expand: Minimum number of episodes required since the last change before any expansion/contraction check occurs.
        :type min_episodes_before_expand: int
        :param dead_zone_radius: Minimum distance from the origin $(0, 0, 0)$ that sampled goal positions must respect (avoids robot spawn area).
        :type dead_zone_radius: float
        :returns: None
        :rtype: None
        """
        self.initial_range = initial_range
        self.target_range = target_range
        self.current_max_distance = initial_range[1]
        self.min_distance = max(initial_range[0], dead_zone_radius)
        self.dead_zone_radius = dead_zone_radius
        
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self.expansion_step = expansion_step
        self.window_size = window_size
        self.min_episodes = min_episodes_before_expand
        
        # Track recent episodes
        self.recent_successes = []
        self.episode_count = 0
        self.episodes_since_last_change = 0

    def save(self, filepath: str) -> None:
        """
        Save the current state of the curriculum manager to a file using pickle.

        The saved state includes the current maximum goal distance, success history, 
        and episode counters, alongside the configuration parameters for verification.

        :param filepath: The full path to the file where the state should be saved.
        :type filepath: str
        :returns: None
        :rtype: None
        """
        state = {
            "current_max_distance": self.current_max_distance,
            "recent_successes": self.recent_successes,
            "episode_count": self.episode_count,
            "episodes_since_last_change": self.episodes_since_last_change,
            "min_distance": self.min_distance,
            "config": {
                "initial_range": self.initial_range,
                "target_range": self.target_range,
                "success_threshold": self.success_threshold,
                "failure_threshold": self.failure_threshold,
                "expansion_step": self.expansion_step,
                "window_size": self.window_size,
                "min_episodes_before_expand": self.min_episodes,
                "dead_zone_radius": self.dead_zone_radius,
            }
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"💾 Curriculum state saved to {filepath}")

    def load(self, filepath: str) -> bool:
        """
        Load the curriculum manager state from a file.

        The function verifies that key configuration parameters match the currently 
        initialized manager to prevent loading incompatible states.

        :param filepath: The full path to the file containing the saved state.
        :type filepath: str
        :returns: True if the state was loaded successfully, False otherwise.
        :rtype: bool
        """
        if not os.path.exists(filepath):
            print(f"Curriculum state file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore state
            self.current_max_distance = state["current_max_distance"]
            self.recent_successes = state["recent_successes"]
            self.episode_count = state["episode_count"]
            self.episodes_since_last_change = state.get("episodes_since_last_change", 0)
            self.min_distance = state["min_distance"]
            # Verify config matches
            loaded_config = state.get("config", {})
            if loaded_config.get("success_threshold") != self.success_threshold:
                print(f"[Curriculum] WARNING: success_threshold changed! "
                      f"Old: {loaded_config.get('success_threshold'):.2f}, New: {self.success_threshold:.2f}")
            
            if loaded_config.get('failure_threshold') != self.failure_threshold:
                print(f"[Curriculum] WARNING: failure_threshold changed! "
                      f"Old: {loaded_config.get('failure_threshold'):.2f}, New: {self.failure_threshold:.2f}")
            
            print(f"[Curriculum] Loaded state: max_distance={self.current_max_distance:.2f}m, "
                  f"episodes={self.episode_count}, success_rate={np.mean(self.recent_successes) if self.recent_successes else 0:.1%}")
            return True
        
        except Exception as e:
            print(f"Failed to load curriculum state: {e}. Starting fresh.")
            return False

    def sample_goal_distance(self) -> float:
        """
        Samples a goal distance (radius) uniformly within the current curriculum range.

        The sampling range is $[\text{min\_distance}, \text{current\_max\_distance}]$.

        :returns: A randomly sampled goal distance in meters.
        :rtype: float
        """
        return np.random.uniform(self.min_distance, self.current_max_distance)
    
    def sample_goal_position(self) -> NDArray[np.float64]:
        """
        Sample a random goal position $(x, y, z)$ within the current curriculum range, avoiding the dead zone.

        The position is sampled by choosing a distance (radius) using :py:meth:`~GoalCurriculumManager.sample_goal_distance` 
        and a random angle, ensuring the distance from the origin $(0, 0)$ is always greater than :py:attr:`self.dead_zone_radius`.

        :returns: A 3D array $[\text{x}, \text{y}, 0.1]$ representing the sampled goal position.
        :rtype: NDArray[np.float64]
        """
        max_attempts = 100

        for _ in range(max_attempts):
            distance = self.sample_goal_distance()
            # Sample a random angle
            angle = np.random.uniform(0, 2 * np.pi)
            # Convert to Cartesian coordinates
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            # Check if outside dead zone
            distance_from_origin = np.sqrt(x**2 + y**2)
            if distance_from_origin >= self.dead_zone_radius:
                return np.array([x, y, 0.1], dtype=np.float64) # Added dtype for explicit type hinting
            
        # Fallback in case of failure to sample (use minimum distance)
        angle = np.random.uniform(0, 2 * np.pi)
        x = self.min_distance * np.cos(angle)
        y = self.min_distance * np.sin(angle)
        return np.array([x, y, 0.1], dtype=np.float64) # Added dtype for explicit type hinting
    
    def record_episode(self, success: bool) -> None:
        """
        Record episode outcome and execute the curriculum logic to update the goal distance range.

        Curriculum adjustment occurs only if the number of episodes since the last change 
        exceeds :py:attr:`self.min_episodes`. 

        * **Expansion:** If success rate $\ge$ :py:attr:`self.success_threshold`, :py:attr:`self.current_max_distance` increases by :py:attr:`self.expansion_step`.
        * **Contraction:** If success rate $<$ :py:attr:`self.failure_threshold`, :py:attr:`self.current_max_distance` decreases by :py:attr:`self.expansion_step`.

        :param success: Boolean indicating if the episode was successful (True) or not (False).
        :type success: bool
        :returns: None
        :rtype: None
        """
        self.recent_successes.append(success)
        self.episode_count += 1
        self.episodes_since_last_change += 1
        
        # Keep only recent window
        if len(self.recent_successes) > self.window_size:
            self.recent_successes.pop(0)
        
        # Only update after minimum episodes since last change
        if self.episodes_since_last_change < self.min_episodes:
            return
        
        # Calculate success rate
        if len(self.recent_successes) >= self.window_size:
            success_rate = np.mean(self.recent_successes)
            
            # Expand curriculum if doing well
            if success_rate >= self.success_threshold:
                old_max = self.current_max_distance
                self.current_max_distance = min(
                    self.current_max_distance + self.expansion_step,
                    self.target_range[1]
                )
                if self.current_max_distance > old_max:
                    print(f"📈 Curriculum expanded! Max distance: {old_max:.2f}m → {self.current_max_distance:.2f}m (Success rate: {success_rate:.1%})")
                    # Reset tracking after expansion
                    self.recent_successes = []
                    self.episodes_since_last_change = 0
            
            # Contract curriculum if struggling
            elif success_rate < self.failure_threshold:
                old_max = self.current_max_distance
                self.current_max_distance = max(
                    self.current_max_distance - self.expansion_step,
                    self.initial_range[1]
                )
                if self.current_max_distance < old_max:
                    print(f"📉 Curriculum contracted! Max distance: {old_max:.2f}m → {self.current_max_distance:.2f}m (Success rate: {success_rate:.1%})")
                    # Reset tracking after contraction
                    self.recent_successes = []
                    self.episodes_since_last_change = 0
    
    def get_curriculum_info(self) -> dict[str, float]:
        """
        Get current curriculum statistics.

        :returns: A dictionary containing the maximum goal distance, current success rate, 
            episode counts, and curriculum progress relative to the target range.
        :rtype: dict[str, float]
        """
        success_rate = np.mean(self.recent_successes) if self.recent_successes else 0.0
        return {
            "curriculum/max_distance": self.current_max_distance,
            "curriculum/success_rate": success_rate,
            "curriculum/episode_count": self.episode_count,
            "curriculum/episodes_since_change": self.episodes_since_last_change,
            "curriculum/progress": (self.current_max_distance - self.initial_range[1]) / (self.target_range[1] - self.initial_range[1])
        }