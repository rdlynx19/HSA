import numpy as np
import pickle, os

class GoalCurriculumManager:
    """
    Manages curriculum for goal positions in the environment.
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
        Initializes the curriculum manager.
        :param initial_range: Initial range for goal positions.
        :param final_range: Final range for goal positions.
        :param success_threshold: Success rate threshold to expand the curriculum.
        :param failure_threshold: Failure rate threshold to contract the curriculum.
        :param expansion_step: Step size for expanding or contracting the curriculum range.
        :param window_size: Number of episodes to consider for success/failure rate.
        :param min_episodes_before_expand: Minimum episodes before allowing expansion.
        :param dead_zone_radius: Minimum distance from origin (0,0) to avoid for goal positions.
        """
        self.initial_range = initial_range
        self.target_range = target_range
        self.current_max_distance = initial_range[1]
        self.min_distance = max(initial_range[0], dead_zone_radius)  # Ensure min > dead zone
        self.dead_zone_radius = dead_zone_radius
        
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self.expansion_step = expansion_step
        self.window_size = window_size
        self.min_episodes = min_episodes_before_expand
        
        # Track recent episodes
        self.recent_successes = []
        self.episode_count = 0

    def save(self, filepath: str):
        """Save the current state of the curriculum manager to a file."""
        state = {
            "current_max_distance": self.current_max_distance,
            "recent_successes": self.recent_successes,
            "episode_count": self.episode_count,
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

    def load(self, filepath: str):
        """Load the curriculum manager state from a file."""
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
        Samples a goal distance within the current curriculum range, avoiding the dead zone.
        """
        return np.random.uniform(self.min_distance, self.current_max_distance)
    
    def sample_goal_position(self) -> np.ndarray:
        """
        Sample a random goal position within the curriculum distance, avoiding the dead zone around origin where robot spawns.
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
                return np.array([x, y, 0.1])
            
        # Fallback in case of failure to sample
        angle = np.random.uniform(0, 2 * np.pi)
        x = self.min_distance * np.cos(angle)
        y = self.min_distance * np.sin(angle)
        return np.array([x, y, 0.1])
    
    def record_episode(self, success: bool):
        """Record episode outcome and potentially update curriculum"""
        self.recent_successes.append(success)
        self.episode_count += 1
        
        # Keep only recent window
        if len(self.recent_successes) > self.window_size:
            self.recent_successes.pop(0)
        
        # Only update after minimum episodes
        if self.episode_count < self.min_episodes:
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
    
    def get_curriculum_info(self) -> dict:
        """Get current curriculum statistics"""
        success_rate = np.mean(self.recent_successes) if self.recent_successes else 0.0
        return {
            "curriculum/max_distance": self.current_max_distance,
            "curriculum/success_rate": success_rate,
            "curriculum/episode_count": self.episode_count,
            "curriculum/progress": (self.current_max_distance - self.initial_range[1]) / (self.target_range[1] - self.initial_range[1])
        }