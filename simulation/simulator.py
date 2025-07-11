"""
Main simulation runner for the Roysambu ward crime simulation.

This module orchestrates the entire simulation, managing agents,
environment, and temporal progression.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime, timedelta

from .agents import AgentManager, CriminalAgent, GuardianAgent, VictimAgent
from .environment import RoysambuEnvironment, TimeManager, WeatherManager


class CrimeSimulator:
    """
    Main simulation engine for crime hotspot modeling in Roysambu ward.
    
    This class coordinates all simulation components and runs the
    agent-based model to generate synthetic crime data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the crime simulator.
        
        Args:
            config: Simulation configuration dictionary
        """
        self.config = config
        self.setup_logging()
        
        # Initialize components
        self.environment = RoysambuEnvironment(config['bounds'])
        self.agent_manager = AgentManager()
        self.time_manager = TimeManager()
        self.weather_manager = WeatherManager()
        
        # Simulation state
        self.current_step = 0
        self.max_steps = config.get('max_steps', 8760)  # Default: 1 year in hours
        self.crime_events = []
        self.is_running = False
        
    def setup_logging(self) -> None:
        """Setup logging for the simulation."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('simulation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_simulation(self) -> None:
        """Initialize all simulation components."""
        self.logger.info("Initializing Roysambu crime simulation...")
        
        # Setup environment
        self.environment.load_geographic_data(self.config.get('data_path', ''))
        self.environment.setup_street_network()
        self.environment.setup_facilities()
        
        # Create agents
        self.agent_manager.create_agents(self.config['agents'])
        
        self.logger.info(f"Simulation initialized with {len(self.agent_manager.criminal_agents)} criminals, "
                        f"{len(self.agent_manager.guardian_agents)} guardians, "
                        f"and {len(self.agent_manager.victim_agents)} potential victims")
        
    def run_simulation(self, save_results: bool = True) -> Dict:
        """
        Run the complete simulation.
        
        Args:
            save_results: Whether to save results to files
            
        Returns:
            Dictionary containing simulation results
        """
        self.logger.info("Starting crime simulation...")
        self.is_running = True
        
        try:
            self.initialize_simulation()
            
            # Main simulation loop
            while self.current_step < self.max_steps and self.is_running:
                self.run_single_step()
                self.current_step += 1
                
                # Log progress every 24 steps (1 day)
                if self.current_step % 24 == 0:
                    self.logger.info(f"Completed day {self.current_step // 24}")
                    
        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted by user")
            self.is_running = False
            
        except Exception as e:
            self.logger.error(f"Simulation error: {str(e)}")
            raise
            
        finally:
            results = self.finalize_simulation()
            if save_results:
                self.save_results(results)
                
        return results
        
    def run_single_step(self) -> None:
        """Execute one time step of the simulation."""
        # Update time and weather
        self.time_manager.advance_time()
        self.weather_manager.update_weather()
        
        # Update all agents
        self.agent_manager.update_all_agents(self.current_step)
        
        # Process crime events for this step
        step_crimes = self.process_crime_events()
        self.crime_events.extend(step_crimes)
        
    def process_crime_events(self) -> List[Dict]:
        """Process and record crime events for current time step."""
        step_crimes = []
        
        # TODO: Implement crime event processing
        # - Agent interactions
        # - Crime commission attempts
        # - Success/failure determination
        # - Event recording
        
        return step_crimes
        
    def finalize_simulation(self) -> Dict:
        """Finalize simulation and prepare results."""
        self.logger.info("Finalizing simulation...")
        
        # Create results summary
        results = {
            'total_steps': self.current_step,
            'total_crimes': len(self.crime_events),
            'crime_events': self.crime_events,
            'agent_statistics': self.agent_manager.get_agent_statistics(),
            'simulation_config': self.config,
            'end_time': datetime.now().isoformat()
        }
        
        return results
        
    def save_results(self, results: Dict) -> None:
        """Save simulation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save crime events as CSV
        if results['crime_events']:
            crime_df = pd.DataFrame(results['crime_events'])
            crime_df.to_csv(f'data/processed/simulated_crimes_{timestamp}.csv', index=False)
            
        # Save full results as JSON
        results_path = f'data/processed/simulation_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        self.logger.info(f"Results saved to {results_path}")
        
    def stop_simulation(self) -> None:
        """Stop the simulation gracefully."""
        self.is_running = False
        self.logger.info("Simulation stop requested")


class SimulationConfig:
    """
    Helper class for managing simulation configuration.
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
        
    @staticmethod
    def get_default_config() -> Dict:
        """Get default configuration for Roysambu simulation."""
        return {
            'bounds': [-1.2200, 36.8900, -1.2000, 36.9100],  # Approx Roysambu bounds
            'max_steps': 8760,  # 1 year in hours
            'agents': {
                'criminals': 50,
                'guardians': 20,
                'victims': 200
            },
            'crime_types': ['theft', 'robbery', 'burglary', 'assault'],
            'facility_types': ['school', 'bar', 'atm', 'market', 'hospital'],
            'output_path': 'data/processed/',
            'visualization_path': 'visualizations/'
        }
        
    @staticmethod
    def save_config(config: Dict, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)


def main():
    """Main function to run the simulation."""
    # Load or create configuration
    config = SimulationConfig.get_default_config()
    
    # Create and run simulator
    simulator = CrimeSimulator(config)
    results = simulator.run_simulation()
    
    print(f"Simulation completed! Generated {len(results['crime_events'])} crime events.")
    

if __name__ == "__main__":
    main()
