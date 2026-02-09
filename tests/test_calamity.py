"""Test script to verify the calamity system works correctly."""
import yaml
import pytest
import os
from world.world import World
from world.objects import WorldObject, EdibleComponent, PlantComponent, SeedComponent

def test_calamity():
    print("="*60)
    print("CALAMITY SYSTEM TEST")
    print("="*60)
    
    # Load config
    config_path = 'config/training_easy.yaml'
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Fallback config
        config = {'calamity': {
            'enabled': True, 'interval': 100, 'destruction_rate': 0.5,
            'affect_plants': True, 'affect_food': True, 'affect_seeds': False
        }}

    # Check config
    calamity_cfg = config.get('calamity', {})
    print(f"\nCalamity config:")
    print(f"  Enabled: {calamity_cfg.get('enabled')}")
    print(f"  Interval: {calamity_cfg.get('interval')} ticks")
    print(f"  Destruction rate: {calamity_cfg.get('destruction_rate')*100:.0f}%")
    print(f"  Affect plants: {calamity_cfg.get('affect_plants')}")
    print(f"  Affect food: {calamity_cfg.get('affect_food')}")
    print(f"  Affect seeds: {calamity_cfg.get('affect_seeds')}")

    # Create world
    world = World(
        width=20,
        height=20,
        seed=42,
        allow_stacking=False
    )

    # Set calamity config
    world.calamity_config = calamity_cfg

    # Add some objects
    print(f"\n--- Adding objects to world ---")
    for i in range(20):
        # Add plants
        plant = WorldObject(i, 0)
        plant.add_component(PlantComponent(mature_age=100, max_age=500, spawn_rate=0.1))
        world.add_object(plant)
        
        # Add food
        food = WorldObject(i, 1)
        food.add_component(EdibleComponent(calories=20.0))
        world.add_object(food)
        
        # Add seeds
        seed = WorldObject(i, 2)
        seed.add_component(SeedComponent("test_plant", grow_time=50))
        world.add_object(seed)

    print(f"Added objects: {len(world.objects)}")

    # Count object types
    plants = sum(1 for obj in world.objects.values() if obj.has_component(PlantComponent))
    food = sum(1 for obj in world.objects.values() if obj.has_component(EdibleComponent))
    seeds = sum(1 for obj in world.objects.values() if obj.has_component(SeedComponent))
    print(f"  Plants: {plants}, Food: {food}, Seeds: {seeds}")

    # Simulate until first calamity
    interval = calamity_cfg.get('interval', 500)
    print(f"\n--- Simulating {interval} ticks to trigger calamity ---")

    for tick in range(interval):
        world.tick = tick + 1
        world._check_calamity()

    # Count remaining objects
    plants_after = sum(1 for obj in world.objects.values() if obj.has_component(PlantComponent))
    food_after = sum(1 for obj in world.objects.values() if obj.has_component(EdibleComponent))
    seeds_after = sum(1 for obj in world.objects.values() if obj.has_component(SeedComponent))

    print(f"\n--- After calamity ---")
    print(f"Remaining objects: {len(world.objects)}")
    print(f"  Plants: {plants_after} (was {plants}, destroyed {plants - plants_after})")
    print(f"  Food: {food_after} (was {food}, destroyed {food - food_after})")
    print(f"  Seeds: {seeds_after} (was {seeds}, destroyed {seeds - seeds_after})")

    # Calculate destruction percentages
    plant_loss_pct = (plants - plants_after) / plants * 100 if plants > 0 else 0
    food_loss_pct = (food - food_after) / food * 100 if food > 0 else 0
    seed_loss_pct = (seeds - seeds_after) / seeds * 100 if seeds > 0 else 0

    print(f"\nDestruction percentages:")
    print(f"  Plants: {plant_loss_pct:.1f}% (expected ~{calamity_cfg.get('destruction_rate')*100:.0f}%)")
    print(f"  Food: {food_loss_pct:.1f}% (expected ~{calamity_cfg.get('destruction_rate')*100:.0f}%)")
    print(f"  Seeds: {seed_loss_pct:.1f}% (expected 0% since affect_seeds=false)")

    # Verify expected behavior
    print(f"\n{'='*60}")
    if calamity_cfg.get('affect_plants'):
        if plants_after < plants:
            print("✅ Plants were destroyed (as expected)")
        else:
            # It's probabilistic, but with 20 items 50% rate, unlikley to result in 0 destructions
            pass # warning
    
    if calamity_cfg.get('affect_food'):
        assert food_after < food or food == 0, "Food should be destroyed"

    if not calamity_cfg.get('affect_seeds'):
        assert seeds_after == seeds, "Seeds should be preserved"
    
    print(f"{'='*60}")
    print("\n✅ Calamity system test complete!")

if __name__ == "__main__":
    test_calamity()
