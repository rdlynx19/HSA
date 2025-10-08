from enum import Enum, auto

class RobotState(Enum):
    """
    Enum for robot states
    """
    IDLE = auto() 
    EXTENDED = auto()
    DRIVING = auto()
    TWISTING = auto() # not implemented
    BENDING = auto() # not implemented
    
