"""
Analyze food availability from the actual simulation logs.
"""

import pandas as pd

print("=" * 60)
print("FOOD ANALYSIS FROM SIMULATION LOGS")
print("=" * 60)

# Load the action logs
df_actions = pd.read_csv('data/logs/agent_actions_20251116_001118.csv')

# Check EAT actions
eat_actions = df_actions[df_actions['action'] == 'EAT']
successful_eats = eat_actions[eat_actions['success'] == True]

print(f"\nTotal EAT attempts: {len(eat_actions)}")
print(f"Successful EAT actions: {len(successful_eats)}")
print(f"Success rate: {len(successful_eats)/len(eat_actions)*100:.1f}%")

# When did agents eat successfully?
if len(successful_eats) > 0:
    print("\n" + "=" * 60)
    print("SUCCESSFUL EATING EVENTS")
    print("=" * 60)
    print(successful_eats[['tick', 'agent_id', 'x_before', 'y_before', 'energy_before', 'energy_after', 'inventory_count', 'message']].head(20).to_string(index=False))
    
    # Check energy gained
    successful_eats['energy_gained'] = successful_eats['energy_after'] - successful_eats['energy_before']
    print("\n" + "=" * 60)
    print("ENERGY GAINED FROM EATING")
    print("=" * 60)
    print(f"Average energy gained: {successful_eats['energy_gained'].mean():.2f}")
    print(f"Min energy gained: {successful_eats['energy_gained'].min():.2f}")
    print(f"Max energy gained: {successful_eats['energy_gained'].max():.2f}")
    
    # Which agents ate successfully?
    print("\n" + "=" * 60)
    print("AGENTS WHO ATE SUCCESSFULLY")
    print("=" * 60)
    agents_who_ate = successful_eats.groupby('agent_id').size().sort_values(ascending=False)
    print(agents_who_ate)

# Check PICK_UP actions (getting berries from ground)
pickup_actions = df_actions[df_actions['action'] == 'PICK_UP']
successful_pickups = pickup_actions[pickup_actions['success'] == True]

print("\n" + "=" * 60)
print("PICK UP ACTIONS")
print("=" * 60)
print(f"Total PICK_UP attempts: {len(pickup_actions)}")
print(f"Successful PICK_UP actions: {len(successful_pickups)}")
if len(pickup_actions) > 0:
    print(f"Success rate: {len(successful_pickups)/len(pickup_actions)*100:.1f}%")

if len(successful_pickups) > 0:
    print("\nSuccessful PICK_UP events (first 20):")
    print(successful_pickups[['tick', 'agent_id', 'x_before', 'y_before', 'inventory_count', 'message']].head(20).to_string(index=False))

# Check if agents with inventory ate from inventory
print("\n" + "=" * 60)
print("INVENTORY-BASED EATING")
print("=" * 60)

# EAT actions where agent had items in inventory
eats_with_inventory = eat_actions[eat_actions['inventory_count'] > 0]
print(f"EAT attempts with items in inventory: {len(eats_with_inventory)}")
print(f"Of these, successful: {eats_with_inventory['success'].sum()}")

# Failed EAT attempts
failed_eats = eat_actions[eat_actions['success'] == False]
print("\n" + "=" * 60)
print("FAILED EAT ATTEMPTS ANALYSIS")
print("=" * 60)
print(f"Failed EAT attempts: {len(failed_eats)}")
print(f"\nInventory count during failed EATs:")
print(failed_eats['inventory_count'].value_counts().sort_index())

# Movement patterns
print("\n" + "=" * 60)
print("MOVEMENT ANALYSIS")
print("=" * 60)
move_actions = df_actions[df_actions['action'] == 'MOVE_FORWARD']
print(f"Total MOVE_FORWARD attempts: {len(move_actions)}")
print(f"Successful moves: {move_actions['success'].sum()}")
if len(move_actions) > 0:
    print(f"Success rate: {move_actions['success'].sum()/len(move_actions)*100:.1f}%")

# Action distribution
print("\n" + "=" * 60)
print("ACTION DISTRIBUTION (ALL AGENTS)")
print("=" * 60)
print(df_actions['action'].value_counts())

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
