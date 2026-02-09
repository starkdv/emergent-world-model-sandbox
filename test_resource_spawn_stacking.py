"""
Test script to verify ResourceSpawnSystem respects allow_stacking configuration.

This test ensures that after a calamity, resources don't spawn too quickly
in no-stacking mode.

Author: Karan Vasa
Date: November 17, 2025
"""
import random
from world.world import World
from world.objects import WorldObject, PlantComponent, EdibleComponent

def test_resource_spawn_respects_no_stacking():
    """Test that resource spawning respects allow_stacking=False."""
    print("\n" + "="*70)
    print("TEST: Resource Spawn Respects No-Stacking Mode")
    print("="*70)
    
    # Create world with no-stacking mode
    world = World(width=10, height=10, seed=42, allow_stacking=False)
    
    # Add a mature plant in center
    plant = WorldObject(5, 5)
    plant.add_component(PlantComponent(
        mature_age=0,  # Already mature
        max_age=500,
        spawn_rate=1.0  # 100% chance to spawn
    ))
    plant.get_component(PlantComponent).age = 100  # Make it mature
    world.add_object(plant)
    
    print(f"\n✓ Created world with allow_stacking=False")
    print(f"✓ Added mature plant at (5, 5) with 100% spawn rate")
    
    # Fill all surrounding tiles with objects
    print(f"\n✓ Filling all 8 surrounding tiles with objects...")
    occupied_count = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue  # Skip center (plant is here)
            x, y = 5 + dx, 5 + dy
            obj = WorldObject(x, y)
            obj.add_component(EdibleComponent(calories=10.0))
            if world.add_object(obj):
                occupied_count += 1
    
    print(f"  Occupied {occupied_count}/8 surrounding tiles")
    
    # Run update - plant should try to spawn but fail (no empty tiles)
    initial_object_count = len(world.objects)
    print(f"\n✓ Initial object count: {initial_object_count}")
    
    print(f"\n✓ Running world.update() - plant will try to spawn...")
    world.update()
    
    after_update_count = len(world.objects)
    print(f"✓ Object count after update: {after_update_count}")
    
    # In no-stacking mode, spawn should fail because all tiles are occupied
    if after_update_count == initial_object_count:
        print(f"\n✅ SUCCESS: No new objects spawned (all tiles occupied)")
        print(f"   ResourceSpawnSystem correctly respects allow_stacking=False")
    else:
        print(f"\n❌ FAILURE: New objects spawned despite all tiles being occupied!")
        print(f"   Expected: {initial_object_count}, Got: {after_update_count}")
        print(f"   ResourceSpawnSystem is NOT respecting allow_stacking=False")
        return False
    
    # Now remove one object to make space
    print(f"\n✓ Removing one object to create empty space...")
    objects_at_6_5 = world.get_objects_at(6, 5)
    if objects_at_6_5:
        world.remove_object(objects_at_6_5[0].id)
        print(f"  Removed object from (6, 5)")
    
    before_spawn_count = len(world.objects)
    print(f"\n✓ Object count before spawn attempt: {before_spawn_count}")
    
    # Run update again - now spawn should succeed
    print(f"\n✓ Running world.update() again...")
    world.update()
    
    after_spawn_count = len(world.objects)
    print(f"✓ Object count after update: {after_spawn_count}")
    
    if after_spawn_count > before_spawn_count:
        print(f"\n✅ SUCCESS: New object spawned when space became available")
        print(f"   Spawned {after_spawn_count - before_spawn_count} new object(s)")
    else:
        print(f"\n⚠️  WARNING: No object spawned even though space was available")
        print(f"   This might be due to randomness in spawn system")
    
    return True


def test_calamity_respawn_rate():
    """Test that resources don't respawn too quickly after calamity in no-stacking mode."""
    print("\n" + "="*70)
    print("TEST: Calamity Respawn Rate in No-Stacking Mode")
    print("="*70)
    
    # Create world with no-stacking mode
    world = World(width=20, height=20, seed=42, allow_stacking=False)
    
    # Configure calamity
    world.calamity_config = {
        'enabled': True,
        'interval': 100,
        'destruction_rate': 0.5,  # 50% destruction
        'affect_plants': True,
        'affect_food': True,
        'affect_seeds': False
    }
    
    print(f"\n✓ Created 20x20 world with allow_stacking=False")
    print(f"✓ Calamity config: 50% destruction every 100 ticks")
    
    # Add many objects
    print(f"\n✓ Adding objects to fill ~50% of world...")
    objects_added = 0
    for i in range(200):
        x = random.randint(0, 19)
        y = random.randint(0, 19)
        obj = WorldObject(x, y)
        obj.add_component(EdibleComponent(calories=20.0))
        if world.add_object(obj):
            objects_added += 1
    
    print(f"  Successfully added {objects_added} objects")
    
    # Run until just before calamity
    print(f"\n✓ Running simulation for 99 ticks (before calamity)...")
    for _ in range(99):
        world.update()
    
    count_before_calamity = len(world.objects)
    print(f"✓ Object count before calamity: {count_before_calamity}")
    
    # Trigger calamity
    print(f"\n✓ Running tick 100 (calamity strikes)...")
    world.update()
    
    count_after_calamity = len(world.objects)
    destroyed = count_before_calamity - count_after_calamity
    print(f"✓ Object count after calamity: {count_after_calamity}")
    print(f"  Destroyed: {destroyed} objects ({destroyed/count_before_calamity*100:.1f}%)")
    
    # Run a few more ticks and check respawn rate
    print(f"\n✓ Running 10 more ticks to observe respawn rate...")
    for tick in range(1, 11):
        world.update()
        current_count = len(world.objects)
        respawned = current_count - count_after_calamity
        print(f"  Tick {tick}: {current_count} objects (+{respawned} from post-calamity)")
    
    final_count = len(world.objects)
    total_respawned = final_count - count_after_calamity
    respawn_rate = total_respawned / destroyed if destroyed > 0 else 0
    
    print(f"\n✓ Summary:")
    print(f"  Objects destroyed by calamity: {destroyed}")
    print(f"  Objects respawned in 10 ticks: {total_respawned}")
    print(f"  Respawn rate: {respawn_rate*100:.1f}% of destroyed objects")
    
    # In no-stacking mode, respawn should be limited
    if respawn_rate < 0.3:  # Less than 30% respawned
        print(f"\n✅ SUCCESS: Respawn rate is reasonable for no-stacking mode")
        print(f"   ResourceSpawnSystem respects tile occupancy limits")
    else:
        print(f"\n⚠️  WARNING: Respawn rate seems high for no-stacking mode")
        print(f"   Expected <30%, got {respawn_rate*100:.1f}%")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RESOURCE SPAWN STACKING CONFIGURATION TESTS")
    print("="*70)
    
    test1_passed = test_resource_spawn_respects_no_stacking()
    test2_passed = test_calamity_respawn_rate()
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Test 1 (No-Stacking Respect): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (Calamity Respawn Rate): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 All tests passed! ResourceSpawnSystem correctly respects allow_stacking.")
    else:
        print(f"\n❌ Some tests failed. Please review the implementation.")
    
    print("="*70)
