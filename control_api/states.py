"""
Robot State Definitions for MuJoCo Control.

This module defines the enumeration used by the `MuJoCoControlInterface` to manage 
the operational state of the Handed Shearing Auxetic (HSA) robot. These states govern 
which complex control routines (like driving or bending) are allowed to be executed.
"""
from enum import Enum, auto

class RobotState(Enum):
    """
    Defines the discrete operational states of the Handed Shearing Auxetic (HSA) robot.

    These states are used by the :py:class:`~MuJoCoControlInterface` and the 
    :py:func:`~require_state` decorator to enforce the correct execution order 
    of control routines (a state machine). 

    Members:
        IDLE: The robot is stationary, at rest, or in a nominal starting configuration (zero control input).
        EXTENDED: The robot is in a fully extended position following an extension control phase.
        DRIVING: The robot is executing a continuous locomotion phase (e.g., constant velocity control).
        TWISTING: The robot is executing a differential twist motion by selectively locking/unlocking constraints.
        BENDING: The robot is executing a differential bending motion by selectively locking/unlocking constraints.
    """
    IDLE = auto() 
    EXTENDED = auto()
    DRIVING = auto()
    TWISTING = auto() 
    BENDING = auto()