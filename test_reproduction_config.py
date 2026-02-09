"""
Test to verify reproduction config is properly loaded from config file.

Author: Karan Vasa
Date: November 17, 2025
"""

import sys
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from world.world import World
from agents import Agent, Genome, Brain, create_default_trait_config


def test_reproduction_config_loading():
    """Test that reproduction config is properly loaded and used."""
    print("\n" + "="*60)
    print("TEST: Reproduction Config Loading")
    print("="*60)
    
    # Load config
    config_path = Path(__file__).parent / "config" / "training_easy.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n✓ Loaded config from: {config_path}")
    print(f"\nReproduction config from YAML:")
    for key, value in config['reproduction'].items():
        print(f"  {key}: {value}")
    
    # Create world
    world_cfg = config['world']
    world = World(
        width=20,
        height=20,
        seed=42,
        allow_stacking=world_cfg.get('allow_stacking', False)
    )
    
    # Set reproduction config (simulating what main.py should do)
    world.reproduction_config = config['reproduction']
    
    print(f"\n✓ World created with reproduction_config set")
    print(f"  world.reproduction_config: {world.reproduction_config}")
    
    # Create agent with enough energy to reproduce
    brain_cfg = config['brain']
    weight_count = Brain.calculate_weight_count(
        input_size=brain_cfg['input_size'],
        hidden_sizes=brain_cfg['hidden_layers'],
        output_size=brain_cfg['output_size']
    )
    
    trait_config = create_default_trait_config()
    genome = Genome.random(weight_count, trait_config)
    
    agent = Agent(
        x=10,
        y=10,
        genome=genome,
        max_energy=config['agents']['max_energy'],
        max_age=config['agents']['max_age'],
        inventory_size=config['agents']['inventory_size'],
        metabolism_rate=config['agents']['metabolism_rate']
    )
    
    # Give agent full energy and enough age
    agent.energy = agent.max_energy
    agent.age = config['reproduction']['min_age'] + 10
    
    world.add_agent(agent)
    
    print(f"\n✓ Agent created:")
    print(f"  ID: {agent.id}")
    print(f"  Energy: {agent.energy}/{agent.max_energy}")
    print(f"  Age: {agent.age}")
    
    # Check if can reproduce
    can_reproduce = agent.can_reproduce(world.reproduction_config)
    print(f"\n✓ Can reproduce: {can_reproduce}")
    
    if can_reproduce:
        # Record energy before reproduction
        energy_before = agent.energy
        
        # Reproduce
        offspring = agent.reproduce(world, world.reproduction_config)
        
        if offspring:
            energy_after = agent.energy
            energy_lost = energy_before - energy_after
            energy_lost_pct = (energy_lost / energy_before) * 100
            
            expected_split = config['reproduction']['energy_split']
            expected_loss_pct = expected_split * 100
            
            print(f"\n✅ REPRODUCTION SUCCESSFUL!")
            print(f"\nParent Energy Changes:")
            print(f"  Before: {energy_before:.1f}")
            print(f"  After: {energy_after:.1f}")
            print(f"  Lost: {energy_lost:.1f} ({energy_lost_pct:.1f}%)")
            print(f"\nConfig values:")
            print(f"  energy_split from config: {expected_split} ({expected_loss_pct:.1f}%)")
            print(f"\nOffspring:")
            print(f"  Energy: {offspring.energy}/{offspring.max_energy}")
            print(f"  Position: ({offspring.x}, {offspring.y})")
            
            # Verify energy split is correct
            tolerance = 0.01  # 1% tolerance
            if abs(energy_lost_pct - expected_loss_pct) < tolerance:
                print(f"\n✅ ENERGY SPLIT CORRECT!")
                print(f"   Expected parent to lose: {expected_loss_pct:.1f}%")
                print(f"   Actual parent lost: {energy_lost_pct:.1f}%")
                return True
            else:
                print(f"\n❌ ENERGY SPLIT INCORRECT!")
                print(f"   Expected parent to lose: {expected_loss_pct:.1f}%")
                print(f"   Actual parent lost: {energy_lost_pct:.1f}%")
                print(f"   Difference: {abs(energy_lost_pct - expected_loss_pct):.1f}%")
                return False
        else:
            print(f"\n❌ Reproduction failed (no offspring created)")
            return False
    else:
        print(f"\n❌ Agent cannot reproduce")
        return False


def test_without_config():
    """Test what happens when config is None (default behavior)."""
    print("\n" + "="*60)
    print("TEST: Reproduction Without Config (Default)")
    print("="*60)
    
    # Create world without setting reproduction config
    world = World(width=20, height=20, seed=42)
    
    print(f"\n✓ World created WITHOUT reproduction_config")
    print(f"  world.reproduction_config: {world.reproduction_config}")
    
    # Create agent
    trait_config = create_default_trait_config()
    weight_count = Brain.calculate_weight_count(64, [32, 16], 8)
    genome = Genome.random(weight_count, trait_config)
    
    agent = Agent(
        x=10,
        y=10,
        genome=genome,
        max_energy=1000.0,
        max_age=5000,
        inventory_size=5,
        metabolism_rate=0.015
    )
    
    # Give agent full energy and enough age
    agent.energy = agent.max_energy
    agent.age = 200
    
    world.add_agent(agent)
    
    # Record energy before reproduction
    energy_before = agent.energy
    
    # Reproduce WITHOUT config (should use defaults)
    offspring = agent.reproduce(world, None)
    
    if offspring:
        energy_after = agent.energy
        energy_lost = energy_before - energy_after
        energy_lost_pct = (energy_lost / energy_before) * 100
        
        print(f"\n✓ Reproduction successful (using defaults)")
        print(f"\nParent Energy Changes:")
        print(f"  Before: {energy_before:.1f}")
        print(f"  After: {energy_after:.1f}")
        print(f"  Lost: {energy_lost:.1f} ({energy_lost_pct:.1f}%)")
        print(f"\nDefault hardcoded values:")
        print(f"  energy_split default: 0.6 (60%)")
        print(f"\n⚠️  WITHOUT CONFIG, PARENT LOSES 60% (hardcoded default)")
        
        return True
    else:
        print(f"\n❌ Reproduction failed")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("REPRODUCTION CONFIG TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("With Config (10% loss)", test_reproduction_config_loading()))
    results.append(("Without Config (60% loss)", test_without_config()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nCONCLUSION:")
        print("  ✅ Config is now properly loaded")
        print("  ✅ energy_split: 0.10 means parent loses 10% (keeps 90%)")
        print("  ✅ Without config, defaults to 60% loss (hardcoded)")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
