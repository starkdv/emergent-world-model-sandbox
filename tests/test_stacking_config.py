"""
Test script to verify the allow_stacking configuration feature.

This script tests both strict mode (allow_stacking=False) and legacy mode
(allow_stacking=True) to ensure proper behavior.

Author: Karan Vasa
Date: November 16, 2025
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from world.world import World
from world.objects import WorldObject, EdibleComponent, SeedComponent
from agents import Agent, Genome, Brain, create_default_trait_config


def test_strict_mode():
    """Test that strict mode prevents object stacking."""
    print("\n" + "="*60)
    print("TEST 1: STRICT MODE (allow_stacking=False)")
    print("="*60)
    
    world = World(
        width=10,
        height=10,
        seed=42,
        allow_stacking=False  # STRICT MODE
    )
    
    print(f"✓ World created with allow_stacking={world.allow_stacking}")
    
    # Try to add multiple objects to same tile
    obj1 = WorldObject(5, 5)
    obj1.add_component(EdibleComponent(calories=20.0))
    
    obj2 = WorldObject(5, 5)  # Same position
    obj2.add_component(EdibleComponent(calories=20.0))
    
    obj3 = WorldObject(5, 5)  # Same position
    obj3.add_component(SeedComponent("test_plant", grow_time=50))
    
    # Add first object - should succeed
    result1 = world.add_object(obj1)
    print(f"\n1st object at (5,5): {'✓ Added' if result1 else '✗ Failed'}")
    
    tile = world.get_tile(5, 5)
    print(f"   Tile (5,5) object count: {len(tile.object_ids)}")
    print(f"   Object position: ({obj1.x}, {obj1.y})")
    
    # Add second object - should be placed nearby
    result2 = world.add_object(obj2)
    print(f"\n2nd object at (5,5): {'✓ Added' if result2 else '✗ Failed'}")
    
    if result2:
        print(f"   Object moved to: ({obj2.x}, {obj2.y})")
        nearby_tile = world.get_tile(obj2.x, obj2.y)
        print(f"   Tile ({obj2.x},{obj2.y}) object count: {len(nearby_tile.object_ids)}")
    
    # Add third object - should be placed nearby
    result3 = world.add_object(obj3)
    print(f"\n3rd object at (5,5): {'✓ Added' if result3 else '✗ Failed'}")
    
    if result3:
        print(f"   Object moved to: ({obj3.x}, {obj3.y})")
        nearby_tile = world.get_tile(obj3.x, obj3.y)
        print(f"   Tile ({obj3.x},{obj3.y}) object count: {len(nearby_tile.object_ids)}")
    
    # Verify no stacking occurred
    tile_check = world.get_tile(5, 5)
    print(f"\n✓ Final check: Tile (5,5) has {len(tile_check.object_ids)} object(s)")
    
    if len(tile_check.object_ids) == 1:
        print("✅ STRICT MODE WORKING: One object per tile enforced!")
    else:
        print("❌ STRICT MODE FAILED: Multiple objects on same tile!")
    
    # Count total objects
    total_objects = len(world.objects)
    print(f"\n✓ Total objects in world: {total_objects}")
    
    return len(tile_check.object_ids) == 1


def test_legacy_mode():
    """Test that legacy mode allows object stacking."""
    print("\n" + "="*60)
    print("TEST 2: LEGACY MODE (allow_stacking=True)")
    print("="*60)
    
    world = World(
        width=10,
        height=10,
        seed=42,
        allow_stacking=True  # LEGACY MODE
    )
    
    print(f"✓ World created with allow_stacking={world.allow_stacking}")
    
    # Try to add multiple objects to same tile
    obj1 = WorldObject(5, 5)
    obj1.add_component(EdibleComponent(calories=20.0))
    
    obj2 = WorldObject(5, 5)  # Same position
    obj2.add_component(EdibleComponent(calories=20.0))
    
    obj3 = WorldObject(5, 5)  # Same position
    obj3.add_component(SeedComponent("test_plant", grow_time=50))
    
    # Add all objects
    result1 = world.add_object(obj1)
    result2 = world.add_object(obj2)
    result3 = world.add_object(obj3)
    
    print(f"\n1st object: {'✓ Added' if result1 else '✗ Failed'}")
    print(f"2nd object: {'✓ Added' if result2 else '✗ Failed'}")
    print(f"3rd object: {'✓ Added' if result3 else '✗ Failed'}")
    
    # Check if they're all on the same tile
    tile = world.get_tile(5, 5)
    print(f"\n✓ Tile (5,5) object count: {len(tile.object_ids)}")
    print(f"   Object 1 position: ({obj1.x}, {obj1.y})")
    print(f"   Object 2 position: ({obj2.x}, {obj2.y})")
    print(f"   Object 3 position: ({obj3.x}, {obj3.y})")
    
    if len(tile.object_ids) == 3:
        print("✅ LEGACY MODE WORKING: Multiple objects stacked on same tile!")
    else:
        print("❌ LEGACY MODE FAILED: Objects not stacking properly!")
    
    return len(tile.object_ids) == 3


def test_agent_drop_action():
    """Test agent drop action respects stacking config."""
    print("\n" + "="*60)
    print("TEST 3: AGENT DROP ACTION (Strict Mode)")
    print("="*60)
    
    world = World(
        width=10,
        height=10,
        seed=42,
        allow_stacking=False
    )
    
    # Create agent with inventory
    brain_cfg = {
        'input_size': 64,
        'hidden_layers': [32, 16],
        'output_size': 8
    }
    weight_count = Brain.calculate_weight_count(
        input_size=brain_cfg['input_size'],
        hidden_sizes=brain_cfg['hidden_layers'],
        output_size=brain_cfg['output_size']
    )
    
    trait_config = create_default_trait_config()
    genome = Genome.random(weight_count, trait_config)
    
    agent = Agent(
        x=5,
        y=5,
        genome=genome,
        max_energy=100.0,
        max_age=1000,
        inventory_size=5,
        metabolism_rate=0.5
    )
    
    world.add_agent(agent)
    
    # Give agent some objects
    obj1 = WorldObject(5, 5)
    obj1.add_component(EdibleComponent(calories=20.0))
    obj2 = WorldObject(5, 5)
    obj2.add_component(SeedComponent("test_plant", grow_time=50))
    
    world.objects[obj1.id] = obj1
    world.objects[obj2.id] = obj2
    agent.inventory.append(obj1.id)
    agent.inventory.append(obj2.id)
    
    print(f"✓ Agent at ({agent.x}, {agent.y}) with {len(agent.inventory)} items")
    
    # Drop first item
    from agents.actions import Action
    result1 = agent._drop(world)
    print(f"\n1st DROP: {result1.message}")
    print(f"   Success: {result1.success}")
    print(f"   Items in inventory: {len(agent.inventory)}")
    
    tile = world.get_tile(agent.x, agent.y)
    print(f"   Items on agent tile: {len(tile.object_ids)}")
    
    # Drop second item - should go nearby since tile occupied
    result2 = agent._drop(world)
    print(f"\n2nd DROP: {result2.message}")
    print(f"   Success: {result2.success}")
    print(f"   Items in inventory: {len(agent.inventory)}")
    
    tile = world.get_tile(agent.x, agent.y)
    print(f"   Items on agent tile: {len(tile.object_ids)}")
    
    # Verify strict mode behavior
    if len(tile.object_ids) <= 1:
        print("\n✅ AGENT DROP WORKING: One object per tile enforced!")
        return True
    else:
        print(f"\n❌ AGENT DROP FAILED: {len(tile.object_ids)} objects on same tile!")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("STACKING CONFIGURATION TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Strict Mode", test_strict_mode()))
    results.append(("Legacy Mode", test_legacy_mode()))
    results.append(("Agent Drop Action", test_agent_drop_action()))
    
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
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
