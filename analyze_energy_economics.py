"""
Energy Economics Analysis

Analyzes the energy flow in the simulation to understand why agents
are dying despite successful food consumption.
"""

import pandas as pd
import os


def find_latest_log():
    """Find the most recent log file."""
    log_dir = "data/logs"
    state_files = [f for f in os.listdir(log_dir) if f.startswith("agent_states_")]
    action_files = [f for f in os.listdir(log_dir) if f.startswith("agent_actions_")]
    
    if not state_files or not action_files:
        return None, None
    
    latest_state = sorted(state_files)[-1]
    latest_action = sorted(action_files)[-1]
    
    return os.path.join(log_dir, latest_state), os.path.join(log_dir, latest_action)


def analyze_energy_economics():
    """Perform detailed energy economics analysis."""
    state_file, action_file = find_latest_log()
    
    if not state_file or not action_file:
        print("❌ No log files found!")
        return
    
    print("=" * 80)
    print("⚡ ENERGY ECONOMICS ANALYSIS")
    print("=" * 80)
    print(f"\n📄 Analyzing:")
    print(f"  States: {os.path.basename(state_file)}")
    print(f"  Actions: {os.path.basename(action_file)}")
    
    # Load data
    states = pd.read_csv(state_file)
    actions = pd.read_csv(action_file)
    
    # Get simulation info
    max_tick = states['tick'].max()
    num_agents = states['agent_id'].nunique()
    
    print(f"\n📊 SIMULATION INFO:")
    print(f"  Duration: {max_tick} ticks")
    print(f"  Agents: {num_agents}")    # Analyze successful eating
    successful_eats = actions[
        (actions['action'] == 'EAT') & 
        (actions['success'] == True)
    ]
    
    # Calculate energy gained from food
    # Note: energy_after - energy_before gives NET change (food - cost)
    # We need to add back the energy_cost to get actual food calories
    successful_eats = successful_eats.copy()
    successful_eats['net_change'] = successful_eats['energy_after'] - successful_eats['energy_before']
    successful_eats['food_calories'] = successful_eats['net_change'] + successful_eats['energy_cost']
    
    total_energy_gained = successful_eats['food_calories'].sum()
    num_eats = len(successful_eats)
    avg_energy_per_eat = total_energy_gained / num_eats if num_eats > 0 else 0
    
    print(f"\n🍎 FOOD CONSUMPTION:")
    print(f"  Successful eats: {num_eats}")
    print(f"  Total energy gained: {total_energy_gained:.1f}")
    print(f"  Average per eat: {avg_energy_per_eat:.1f}")
    
    # Analyze energy expenditure
    print(f"\n⚡ ENERGY EXPENDITURE:")
    
    # Calculate total energy spent
    # Get first and last energy for each agent
    agent_energy_analysis = []
    
    for agent_id in states['agent_id'].unique():
        agent_states = states[states['agent_id'] == agent_id].sort_values('tick')
        
        if len(agent_states) == 0:
            continue
        
        starting_energy = agent_states.iloc[0]['energy']
        final_energy = agent_states.iloc[-1]['energy']
        ticks_alive = len(agent_states)
        death_tick = agent_states.iloc[-1]['tick']
          # Total energy available = starting + gained from food
        agent_eats = successful_eats[successful_eats['agent_id'] == agent_id]
        energy_from_food = agent_eats['food_calories'].sum()
        total_available = starting_energy + energy_from_food
        
        # Energy spent = available - final
        energy_spent = total_available - final_energy
        
        # Metabolic cost (rough estimate)
        agent_actions = actions[actions['agent_id'] == agent_id]
        metabolism_rate = agent_states.iloc[0]['metabolism_rate'] if 'metabolism_rate' in agent_states.columns else 0.03
        metabolic_cost = ticks_alive * metabolism_rate
        
        # Action costs (estimated)
        num_moves = len(agent_actions[agent_actions['action'] == 'MOVE_FORWARD'])
        move_cost = num_moves * 1.0  # Assuming 1 energy per move
        
        other_actions = len(agent_actions[~agent_actions['action'].isin(['MOVE_FORWARD', 'WAIT', 'TURN_LEFT', 'TURN_RIGHT'])])
        other_cost = other_actions * 0.5  # Assuming 0.5 energy per action
        
        agent_energy_analysis.append({
            'agent_id': agent_id,
            'starting_energy': starting_energy,
            'energy_from_food': energy_from_food,
            'total_available': total_available,
            'final_energy': final_energy,
            'energy_spent': energy_spent,
            'ticks_alive': ticks_alive,
            'death_tick': death_tick,
            'num_eats': len(agent_eats),
            'metabolism_rate': metabolism_rate,
            'metabolic_cost_estimate': metabolic_cost,
            'move_cost_estimate': move_cost,
            'other_cost_estimate': other_cost,
            'total_cost_estimate': metabolic_cost + move_cost + other_cost,
            'num_moves': num_moves
        })
    
    df = pd.DataFrame(agent_energy_analysis)
    
    # Overall statistics
    print(f"  Average starting energy: {df['starting_energy'].mean():.1f}")
    print(f"  Average energy from food: {df['energy_from_food'].mean():.1f}")
    print(f"  Average total available: {df['total_available'].mean():.1f}")
    print(f"  Average energy spent: {df['energy_spent'].mean():.1f}")
    print(f"  Average ticks alive: {df['ticks_alive'].mean():.1f}")
    
    print(f"\n💰 COST BREAKDOWN (estimates):")
    print(f"  Average metabolic cost: {df['metabolic_cost_estimate'].mean():.1f}")
    print(f"  Average movement cost: {df['move_cost_estimate'].mean():.1f}")
    print(f"  Average other actions cost: {df['other_cost_estimate'].mean():.1f}")
    print(f"  Average total cost: {df['total_cost_estimate'].mean():.1f}")
    
    print(f"\n📈 ENERGY BALANCE:")
    avg_available = df['total_available'].mean()
    avg_spent = df['energy_spent'].mean()
    balance = avg_available - avg_spent
    
    print(f"  Average available: {avg_available:.1f}")
    print(f"  Average spent: {avg_spent:.1f}")
    print(f"  Balance: {balance:.1f}")
    
    # Calculate burn rate
    avg_ticks = df['ticks_alive'].mean()
    avg_burn_rate = avg_spent / avg_ticks if avg_ticks > 0 else 0
    
    print(f"\n🔥 BURN RATE:")
    print(f"  Average: {avg_burn_rate:.3f} energy/tick")
    print(f"  Average metabolism rate: {df['metabolism_rate'].mean():.3f}")
    
    # Calculate how much energy needed to reach 2000 ticks
    target_ticks = 2000
    energy_needed_for_target = avg_burn_rate * target_ticks
    energy_deficit = energy_needed_for_target - avg_available
    
    print(f"\n🎯 TARGET ANALYSIS (2000 ticks):")
    print(f"  Energy needed: {energy_needed_for_target:.1f}")
    print(f"  Energy available: {avg_available:.1f}")
    print(f"  Deficit: {energy_deficit:.1f}")
    
    # How many more berries needed?
    berries_needed = energy_deficit / avg_energy_per_eat if avg_energy_per_eat > 0 else 0
    
    print(f"\n🍎 TO REACH TARGET:")
    print(f"  Additional berries needed: {berries_needed:.1f}")
    print(f"  Current avg eats: {df['num_eats'].mean():.1f}")
    print(f"  Required avg eats: {df['num_eats'].mean() + berries_needed:.1f}")
    
    # Per-agent breakdown
    print(f"\n" + "=" * 80)
    print("📋 PER-AGENT BREAKDOWN:")
    print("=" * 80)
    print(f"{'Agent':<8} {'Start':<8} {'Food':<8} {'Spent':<8} {'Alive':<8} {'Eats':<8} {'Moves':<8} {'Metab':<8}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['agent_id']:<8} "
              f"{row['starting_energy']:<8.1f} "
              f"{row['energy_from_food']:<8.1f} "
              f"{row['energy_spent']:<8.1f} "
              f"{row['ticks_alive']:<8.0f} "
              f"{row['num_eats']:<8.0f} "
              f"{row['num_moves']:<8.0f} "
              f"{row['metabolism_rate']:<8.3f}")
    
    # Recommendations
    print(f"\n" + "=" * 80)
    print("💡 RECOMMENDATIONS:")
    print("=" * 80)
    
    if energy_deficit > 0:
        print(f"\n⚠️  ENERGY DEFICIT: {energy_deficit:.1f}")
        print(f"\nThe agents need approximately {berries_needed:.1f} more successful eats")
        print(f"to reach the 2000 tick target.\n")
        
        # Option 1: More starting energy
        additional_starting = energy_deficit
        print(f"Option 1: INCREASE STARTING ENERGY")
        print(f"  Current: {df['starting_energy'].mean():.1f}")
        print(f"  Suggested: {df['starting_energy'].mean() + additional_starting:.1f}")
        print(f"  Change: +{additional_starting:.1f}")
        
        # Option 2: More food
        current_eats = df['num_eats'].mean()
        target_eats = current_eats + berries_needed
        eat_success_rate = 0.032  # From analyze_v3_1.py
        eat_attempts = successful_eats['agent_id'].count() / num_agents  # Total attempts per agent
        
        # If we need more eats, we need more berries or higher success rate
        additional_berries_needed = berries_needed / eat_success_rate
        
        print(f"\nOption 2: ADD MORE BERRIES")
        print(f"  Current berries in world: 100")
        print(f"  Additional needed: {additional_berries_needed:.0f}")
        print(f"  Suggested: {100 + additional_berries_needed:.0f}")
        
        # Option 3: Reduce metabolism
        target_metabolism = (avg_available / target_ticks)
        print(f"\nOption 3: REDUCE METABOLISM")
        print(f"  Current avg: {df['metabolism_rate'].mean():.3f}")
        print(f"  Suggested: {target_metabolism:.3f}")
        print(f"  Change: {target_metabolism - df['metabolism_rate'].mean():.3f}")
        
        # Option 4: More calories per berry
        calories_per_berry = avg_energy_per_eat
        target_calories = calories_per_berry * (target_eats / current_eats) if current_eats > 0 else calories_per_berry
        
        print(f"\nOption 4: INCREASE BERRY CALORIES")
        print(f"  Current: {calories_per_berry:.1f}")
        print(f"  Suggested: {target_calories:.1f}")
        print(f"  Change: +{target_calories - calories_per_berry:.1f}")
        
    else:
        print(f"\n✅ ENERGY SURPLUS: {-energy_deficit:.1f}")
        print(f"\nAgents have enough energy but are dying for other reasons.")
        print(f"Possible issues:")
        print(f"  - High variance in metabolism rates")
        print(f"  - Uneven food distribution")
        print(f"  - Bad luck in exploration")
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_energy_economics()
