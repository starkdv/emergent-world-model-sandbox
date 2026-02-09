"""
Quick analysis for V3.1 test results.
Analyzes the most recent log file (supports both old and new formats).

Supports:
- Old format: agent_actions_*.csv (from AgentLogger)
- New format: transitions_*.csv (from WorldModelLogger)
"""

import pandas as pd
import glob
import os

# Find most recent log file (check both formats)
log_dir = 'data/logs'

# Look for both old and new format files
action_files = glob.glob(os.path.join(log_dir, 'agent_actions_*.csv'))
transition_files = glob.glob(os.path.join(log_dir, 'transitions_*.csv'))

all_files = action_files + transition_files

if not all_files:
    print("❌ No log files found!")
    print(f"  Searched in: {log_dir}")
    print(f"  Looking for: agent_actions_*.csv or transitions_*.csv")
    exit(1)

latest_file = max(all_files, key=os.path.getctime)
is_new_format = 'transitions_' in os.path.basename(latest_file)

print(f"📊 Analyzing: {latest_file}")
print(f"   Format: {'WorldModelLogger (new)' if is_new_format else 'AgentLogger (old)'}")
print("=" * 70)

# Load data (with low_memory=False to avoid mixed type warnings)
df = pd.read_csv(latest_file, low_memory=False)

# Normalize column names for compatibility
if is_new_format:
    # New format uses 'action' directly (already a string name)
    # The new format stores success as int (0/1), convert to bool
    if 'success' in df.columns:
        df['success'] = df['success'].astype(bool)
    # Fill NaN in death_reason with empty string
    if 'death_reason' in df.columns:
        df['death_reason'] = df['death_reason'].fillna('')
else:
    # Old format - columns should already be compatible
    pass

# Basic stats
total_actions = len(df)
total_ticks = df['tick'].max()
num_agents = df['agent_id'].nunique()

print(f"\n📈 SIMULATION SUMMARY")
print(f"  Total ticks: {total_ticks}")
print(f"  Total actions: {total_actions:,}")
print(f"  Number of agents: {num_agents}")

# New format extras
if is_new_format:
    # Check for episodes and world states files
    episodes_file = latest_file.replace('transitions_', 'episodes_')
    world_states_file = latest_file.replace('transitions_', 'world_states_')
    
    if os.path.exists(episodes_file):
        episodes_df = pd.read_csv(episodes_file)
        if len(episodes_df) > 0:
            print(f"\n📋 EPISODE SUMMARY (from episodes file)")
            print(f"  Total episodes: {len(episodes_df)}")
            print(f"  Avg duration: {episodes_df['duration'].mean():.1f} ticks")
            print(f"  Avg reward: {episodes_df['total_reward'].mean():.2f}")
            print(f"  Avg successful eats: {episodes_df['successful_eats'].mean():.1f}")
            print(f"  Avg tiles explored: {episodes_df['tiles_explored'].mean():.1f}")
    
    if os.path.exists(world_states_file):
        world_df = pd.read_csv(world_states_file)
        if len(world_df) > 0:
            print(f"\n🌍 WORLD STATE SUMMARY")
            print(f"  Avg food count: {world_df['total_food'].mean():.1f}")
            print(f"  Avg plants: {world_df['total_plants'].mean():.1f}")
            print(f"  Avg agent energy: {world_df['avg_agent_energy'].mean():.1f}")
            print(f"  Max agent age reached: {world_df['max_agent_age'].max()}")
    
    # Reward analysis (new format only)
    if 'reward' in df.columns:
        print(f"\n💰 REWARD ANALYSIS")
        print(f"  Total reward: {df['reward'].sum():.2f}")
        print(f"  Avg reward per action: {df['reward'].mean():.4f}")
        print(f"  Max reward: {df['reward'].max():.2f}")
        print(f"  Min reward: {df['reward'].min():.2f}")
        
        # Reward by action type
        print(f"\n  Reward by action:")
        reward_by_action = df.groupby('action')['reward'].agg(['mean', 'sum', 'count'])
        for action, row in reward_by_action.iterrows():
            print(f"    {action:15s}: avg={row['mean']:+.3f}, total={row['sum']:+.1f}, count={int(row['count'])}")

# Action distribution
print(f"\n🎯 ACTION DISTRIBUTION")
action_counts = df['action'].value_counts()
for action, count in action_counts.items():
    pct = (count / total_actions) * 100
    print(f"  {action:15s}: {count:5d} ({pct:5.1f}%)")

# Success rates by action
print(f"\n✅ SUCCESS RATES")
for action in df['action'].unique():
    action_df = df[df['action'] == action]
    success_rate = (action_df['success'].sum() / len(action_df)) * 100
    print(f"  {action:15s}: {success_rate:5.1f}%")

# EAT action analysis
eat_actions = df[df['action'] == 'EAT']
if len(eat_actions) > 0:
    eat_success = eat_actions['success'].sum()
    eat_attempts = len(eat_actions)
    eat_success_rate = (eat_success / eat_attempts) * 100
    
    print(f"\n🍎 FOOD CONSUMPTION")
    print(f"  EAT attempts: {eat_attempts}")
    print(f"  Successful: {eat_success}")
    print(f"  Success rate: {eat_success_rate:.1f}%")
    
    # Show successful eating events
    if eat_success > 0:
        successful_eats = eat_actions[eat_actions['success'] == True]
        print(f"\n  Successful eating events:")
        for _, row in successful_eats.head(10).iterrows():
            if 'message' in df.columns:
                print(f"    Tick {row['tick']:4d}, Agent {row['agent_id']}: {row['message']}")
            else:
                # New format doesn't have message, show energy change instead
                if 'energy' in df.columns and 'energy_next' in df.columns:
                    energy_gain = row['energy_next'] - row['energy']
                    print(f"    Tick {row['tick']:4d}, Agent {row['agent_id']}: energy +{energy_gain:.1f}")
                else:
                    print(f"    Tick {row['tick']:4d}, Agent {row['agent_id']}")

# Movement analysis
move_actions = df[df['action'] == 'MOVE_FORWARD']
if len(move_actions) > 0:
    move_success = move_actions['success'].sum()
    move_attempts = len(move_actions)
    move_success_rate = (move_success / move_attempts) * 100
    
    print(f"\n🚶 MOVEMENT")
    print(f"  MOVE attempts: {move_attempts}")
    print(f"  Successful: {move_success}")
    print(f"  Success rate: {move_success_rate:.1f}%")

# WAIT analysis
wait_actions = df[df['action'] == 'WAIT']
wait_count = len(wait_actions)
wait_pct = (wait_count / total_actions) * 100

print(f"\n⏸️  WAITING BEHAVIOR")
print(f"  WAIT actions: {wait_count}")
print(f"  Percentage: {wait_pct:.1f}%")

# Compare to targets
print(f"\n🎯 V3.1 TARGET COMPARISON")
print(f"  WAIT actions:")
print(f"    Actual: {wait_pct:.1f}%")
print(f"    Target: 32-35%")
if 32 <= wait_pct <= 35:
    print(f"    Status: ✅ WITHIN TARGET")
elif wait_pct < 32:
    print(f"    Status: ⚠️  TOO AGGRESSIVE (decrease exploration bonus)")
else:
    print(f"    Status: ⚠️  TOO PASSIVE (increase exploration bonus)")

if len(eat_actions) > 0:
    print(f"  EAT success rate:")
    print(f"    Actual: {eat_success_rate:.1f}%")
    print(f"    Target: 4-5%")
    if 4 <= eat_success_rate <= 5:
        print(f"    Status: ✅ WITHIN TARGET")
    elif eat_success_rate < 3:
        print(f"    Status: ⚠️  TOO LOW (agents not finding/eating food)")
    else:
        print(f"    Status: ✅ ABOVE TARGET (good!)")

print(f"  Survival time:")
print(f"    Actual: {total_ticks} ticks")
print(f"    Target: >1500 ticks")
if total_ticks >= 1500:
    print(f"    Status: ✅ TARGET MET")
else:
    print(f"    Status: ⚠️  BELOW TARGET")

# Overall verdict
print(f"\n🏆 OVERALL VERDICT")
within_wait_target = 32 <= wait_pct <= 35
good_eat_rate = eat_success_rate >= 3 if len(eat_actions) > 0 else False
good_survival = total_ticks >= 1500

if within_wait_target and good_eat_rate and good_survival:
    print(f"  ✅ SUCCESS! V3.1 parameters are well-tuned!")
    print(f"  - Balanced exploration vs exploitation")
    print(f"  - Good food finding and eating")
    print(f"  - Strong survival time")
elif within_wait_target and good_eat_rate:
    print(f"  ⚠️  PARTIAL SUCCESS - Good behavior but short survival")
    print(f"  - Consider longer test runs")
elif wait_pct < 30:
    print(f"  ❌ TOO AGGRESSIVE - Reduce exploration bonus to +0.10")
elif wait_pct > 38:
    print(f"  ❌ TOO PASSIVE - Increase exploration bonus to +0.20")
else:
    print(f"  ⚠️  MIXED RESULTS - Analyze detailed logs")

# Additional analysis for new format
if is_new_format:
    print(f"\n" + "=" * 70)
    print("🧠 WORLD MODEL TRAINING DATA ANALYSIS")
    print("=" * 70)
    
    # Energy dynamics
    if 'energy' in df.columns and 'energy_next' in df.columns:
        print(f"\n⚡ ENERGY DYNAMICS")
        df['energy_delta'] = df['energy_next'] - df['energy']
        print(f"  Avg energy change per action: {df['energy_delta'].mean():.3f}")
        print(f"  Total energy gained: {df[df['energy_delta'] > 0]['energy_delta'].sum():.1f}")
        print(f"  Total energy lost: {df[df['energy_delta'] < 0]['energy_delta'].sum():.1f}")
        
        # Energy change by action
        print(f"\n  Energy change by action:")
        energy_by_action = df.groupby('action')['energy_delta'].mean()
        for action, delta in energy_by_action.items():
            symbol = "+" if delta > 0 else ""
            print(f"    {action:15s}: {symbol}{delta:.3f}")
    
    # Observation vector analysis
    obs_cols = [c for c in df.columns if c.startswith('obs_') and not c.startswith('obs_next')]
    if obs_cols:
        print(f"\n👁️ OBSERVATION ANALYSIS")
        print(f"  Observation dimensions: {len(obs_cols)}")
        
        # Check observation variance (low variance = limited exploration)
        obs_data = df[obs_cols].values
        obs_variance = obs_data.var(axis=0).mean()
        print(f"  Avg observation variance: {obs_variance:.4f}")
        
        # Energy observation (first feature)
        if 'obs_0' in df.columns:
            print(f"  Energy observation (obs_0):")
            print(f"    Mean: {df['obs_0'].mean():.3f}")
            print(f"    Std:  {df['obs_0'].std():.3f}")
            print(f"    Range: {df['obs_0'].min():.3f} - {df['obs_0'].max():.3f}")
    
    # Tile analysis
    if 'tile_has_food' in df.columns:
        print(f"\n🗺️ TILE ANALYSIS")
        food_encounters = df['tile_has_food'].sum()
        print(f"  Actions on food tiles: {food_encounters} ({food_encounters/len(df)*100:.1f}%)")
        
        if 'tile_fertility' in df.columns:
            print(f"  Avg tile fertility: {df['tile_fertility'].mean():.3f}")
        if 'tile_moisture' in df.columns:
            print(f"  Avg tile moisture: {df['tile_moisture'].mean():.3f}")
    
    # Done/death analysis
    if 'done' in df.columns:
        deaths = df['done'].sum()
        print(f"\n💀 TERMINATION ANALYSIS")
        print(f"  Total deaths logged: {deaths}")
        if deaths > 0 and 'death_reason' in df.columns:
            death_reasons = df[df['done'] == True]['death_reason'].value_counts()
            print(f"  Death reasons:")
            for reason, count in death_reasons.items():
                if reason:  # Skip empty strings
                    print(f"    {reason}: {count}")
    
    # Data quality check
    print(f"\n📊 DATA QUALITY")
    print(f"  Total transitions: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    
    # Count actual missing values (excluding death_reason which is intentionally empty for non-death rows)
    missing_count = df.drop(columns=['death_reason'], errors='ignore').isnull().sum().sum()
    print(f"  Missing values: {missing_count}")
    
    # Check for any problematic columns
    cols_with_nulls = df.isnull().sum()
    cols_with_nulls = cols_with_nulls[cols_with_nulls > 0]
    if len(cols_with_nulls) > 0:
        print(f"  Columns with nulls:")
        for col, count in cols_with_nulls.head(5).items():
            if col != 'death_reason':  # Skip death_reason (expected to have nulls)
                print(f"    {col}: {count} nulls")
    
    # Estimate training data size
    obs_next_cols = [c for c in df.columns if c.startswith('obs_next_')]
    if obs_cols and obs_next_cols:
        print(f"\n🎓 WORLD MODEL TRAINING READINESS")
        print(f"  State dimensions: {len(obs_cols)}")
        print(f"  Next state dimensions: {len(obs_next_cols)}")
        print(f"  Action classes: {df['action'].nunique()}")
        print(f"  Transitions available: {len(df):,}")
        
        min_samples = 10000
        if len(df) >= min_samples:
            print(f"  ✅ Sufficient data for initial training (>{min_samples})")
        else:
            print(f"  ⚠️  Need more data ({len(df)}/{min_samples} samples)")
            print(f"     Run simulation longer to collect more transitions")

print("=" * 70)
