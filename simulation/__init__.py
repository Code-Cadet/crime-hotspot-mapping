"""
Simulation package for agent-based crime modeling.

This package provides tools for simulating crime patterns in Roysambu ward
using agent-based modeling approaches.
"""

__version__ = "0.1.0"
__author__ = "Crime Hotspot Analysis Team"

from .simulator import CrimeSimulator, SimulationConfig
from .agents import CriminalAgent, GuardianAgent, VictimAgent, AgentManager
from .environment import RoysambuEnvironment, TimeManager, WeatherManager

__all__ = [
    'CrimeSimulator',
    'SimulationConfig', 
    'CriminalAgent',
    'GuardianAgent',
    'VictimAgent',
    'AgentManager',
    'RoysambuEnvironment',
    'TimeManager',
    'WeatherManager'
]
