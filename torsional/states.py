from enum import Enum, auto

class RobotState(Enum):
    """
    Enum for robot states
    """
    IDLE = auto()
    EXTENDED = auto()
    DRIVE = auto()