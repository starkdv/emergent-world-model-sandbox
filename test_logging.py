"""
Test script for CSV agent logging functionality.

This script runs a short simulation with logging enabled to verify
that the CSV logging system works correctly.

Author: Karan Vasa
Date: November 14, 2025
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from world.world import World
from agents.agent import Agent
from agents.brain import Brain
from agents.genome import Genome, create_default_trait_config
from utils.data import AgentLogger

def test_agent_logging():
    """Test the agent logging system with a simple simulation."""
    
    print("="*60)
    print("AGENT LOGGING TEST")
    print("="*60)
    
    # Create a small world
    print("\n1. Creating world (20x20)...")
    world = World(width=20, height=20, seed=42)
    print(f"   World created with {len(world.objects)} objects")
    
    # Initialize logger
    print("\n2. Initializing logger...")
    logger = AgentLogger(output_dir="data/logs", log_every_n_ticks=1)
    Agent.logger = logger
    print("   Logger initialized")
    
    # Create some agents
    print("\n3. Creating 5 agents...")
    trait_config = create_default_trait_config()
    weight_count = Brain.calculate_weight_count()
    
    agents_created = 0
    for i in range(50):  # Try up to 50 times to place 5 agents
        x = (i * 3) % world.width
        y = (i * 2) % world.height
        tile = world.get_tile(x, y)
        
        if tile and tile.is_passable():
            genome = Genome.random(weight_count, trait_config)
            agent = Agent(x=x, y=y, genome=genome, max_energy=100.0)
            world.add_agent(agent)
            agents_created += 1
            print(f"   Agent {agent.id} created at ({x}, {y})")
            
            if agents_created >= 5:
                break
    
    print(f"\n4. Running simulation for 10 ticks...")
    for tick in range(10):
        world.update()
        print(f"   Tick {tick + 1}/10 - {len(world.agents)} agents alive")
    
    print(f"\n5. Closing logger...")
    logger.close()
    Agent.logger = None
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print(f"\nCheck the following files:")
    print(f"  - {logger.action_file}")
    print(f"  - {logger.state_file}")
    print("\nThese CSV files contain:")
    print("  - agent_actions_*.csv: Every action taken by each agent")
    print("  - agent_states_*.csv: Agent state snapshots at each tick")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(test_agent_logging())
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
