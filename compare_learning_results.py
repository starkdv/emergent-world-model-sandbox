"""
Compare learning results before and after reward improvements.
"""

import pandas as pd
from pathlib import Path

print("=" * 70)
print("LEARNING IMPROVEMENT COMPARISON")
print("=" * 70)

# Log files
old_log = 'data/logs/agent_actions_20251116_003549.csv'
new_log = 'data/logs/agent_actions_20251116_004802.csv'

# Check if files exist
old_exists = Path(old_log).exists()
new_exists = Path(new_log).exists()

if not old_exists:
    print(f"\n❌ Old log not found: {old_log}")
    print("This should be the baseline test from before improvements.")
    
if not new_exists:
    print(f"\n❌ New log not found: {new_log}")
    print("Run the simulation to generate new data:")
    print("  python main.py --config config/training_easy.yaml --learning --log --gui")
    exit(1)

# Load data
print("\n📊 Loading data...")
df_old = pd.read_csv(old_log) if old_exists else pd.DataFrame()
df_new = pd.read_csv(new_log)

if len(df_new) == 0:
    print("\n❌ New log is empty! The simulation didn't run.")
    print("\nTo test improvements:")
    print("  1. Run: python main.py --config config/training_easy.yaml --learning --log --gui")
    print("  2. Let it run for 500-1000 ticks")
    print("  3. Press ESC to exit")
    print("  4. Run this script again")
    exit(1)

print(f"✅ Old data: {len(df_old)} actions")
print(f"✅ New data: {len(df_new)} actions")

# Analysis
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

if old_exists and len(df_old) > 0:
    # Action distribution comparison
    print("\n📊 ACTION DISTRIBUTION:")
    print(f"{'Action':<15} {'Old %':<10} {'New %':<10} {'Change':<10}")
    print("-" * 50)
    
    old_actions = df_old['action'].value_counts()
    new_actions = df_new['action'].value_counts()
    
    old_total = len(df_old)
    new_total = len(df_new)
    
    all_actions = set(old_actions.index) | set(new_actions.index)
    
    for action in sorted(all_actions):
        old_pct = (old_actions.get(action, 0) / old_total * 100) if old_total > 0 else 0
        new_pct = (new_actions.get(action, 0) / new_total * 100) if new_total > 0 else 0
        change = new_pct - old_pct
        change_str = f"{change:+.1f}%"
        
        # Highlight important changes
        if action == 'WAIT' and change < -5:
            change_str += " ✅"
        elif action == 'MOVE_FORWARD' and change > 5:
            change_str += " ✅"
        elif action == 'EAT' and change > 2:
            change_str += " ✅"
        
        print(f"{action:<15} {old_pct:>6.1f}%   {new_pct:>6.1f}%   {change_str:<10}")
    
    # EAT success rate comparison
    print("\n🍎 EATING PERFORMANCE:")
    print(f"{'Metric':<30} {'Old':<15} {'New':<15} {'Change':<10}")
    print("-" * 70)
    
    old_eats = df_old[df_old['action'] == 'EAT']
    new_eats = df_new[df_new['action'] == 'EAT']
    
    old_eat_success = (old_eats['success'].sum() / len(old_eats) * 100) if len(old_eats) > 0 else 0
    new_eat_success = (new_eats['success'].sum() / len(new_eats) * 100) if len(new_eats) > 0 else 0
    
    old_successful = old_eats['success'].sum()
    new_successful = new_eats['success'].sum()
    
    print(f"{'Total EAT attempts':<30} {len(old_eats):<15} {len(new_eats):<15} {new_eats.shape[0] - old_eats.shape[0]:+d}")
    print(f"{'Successful EATs':<30} {old_successful:<15} {new_successful:<15} {new_successful - old_successful:+d}")
    print(f"{'Success rate':<30} {old_eat_success:<14.1f}% {new_eat_success:<14.1f}% {new_eat_success - old_eat_success:+.1f}%")
    
    # Movement comparison
    print("\n🚀 MOVEMENT BEHAVIOR:")
    print(f"{'Metric':<30} {'Old':<15} {'New':<15} {'Change':<10}")
    print("-" * 70)
    
    old_moves = df_old[df_old['action'] == 'MOVE_FORWARD']
    new_moves = df_new[df_new['action'] == 'MOVE_FORWARD']
    
    old_move_pct = (len(old_moves) / old_total * 100) if old_total > 0 else 0
    new_move_pct = (len(new_moves) / new_total * 100) if new_total > 0 else 0
    
    old_move_success = (old_moves['success'].sum() / len(old_moves) * 100) if len(old_moves) > 0 else 0
    new_move_success = (new_moves['success'].sum() / len(new_moves) * 100) if len(new_moves) > 0 else 0
    
    print(f"{'MOVE attempts':<30} {len(old_moves):<15} {len(new_moves):<15} {len(new_moves) - len(old_moves):+d}")
    print(f"{'MOVE % of all actions':<30} {old_move_pct:<14.1f}% {new_move_pct:<14.1f}% {new_move_pct - old_move_pct:+.1f}%")
    print(f"{'MOVE success rate':<30} {old_move_success:<14.1f}% {new_move_success:<14.1f}% {new_move_success - old_move_success:+.1f}%")
    
    # Simulation length
    print("\n⏱️ SIMULATION DURATION:")
    old_ticks = df_old['tick'].max() if len(df_old) > 0 else 0
    new_ticks = df_new['tick'].max() if len(df_new) > 0 else 0
    print(f"  Old: {old_ticks} ticks")
    print(f"  New: {new_ticks} ticks")
    print(f"  Change: {new_ticks - old_ticks:+d} ticks")
      # Overall assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)
    
    improvements = []
    regressions = []
    
    # Check if WAIT percentage decreased
    old_wait_pct = (old_actions.get('WAIT', 0) / old_total * 100) if old_total > 0 else 0
    new_wait_pct = (new_actions.get('WAIT', 0) / new_total * 100) if new_total > 0 else 0
    if new_wait_pct < old_wait_pct:
        improvements.append("✅ Less waiting (more active)")
    
    if new_move_pct > old_move_pct:
        improvements.append("✅ More movement")
    
    if new_eat_success > old_eat_success:
        improvements.append("✅ Better eating success rate")
    
    if new_successful > old_successful:
        improvements.append("✅ More successful eating events")
    
    if len(improvements) > 0:
        print("\n🎉 IMPROVEMENTS:")
        for imp in improvements:
            print(f"  {imp}")
    
    if len(regressions) > 0:
        print("\n⚠️ REGRESSIONS:")
        for reg in regressions:
            print(f"  {reg}")
    
    if len(improvements) >= 2:
        print("\n✅ RESULT: Reward improvements are WORKING!")
    elif len(improvements) >= 1:
        print("\n⚡ RESULT: Some improvements, may need more tuning")
    else:
        print("\n❌ RESULT: No significant improvement, needs investigation")

else:
    print("\n⚠️ No baseline data to compare against.")
    print("Showing current results only:\n")
    
    print(f"📊 Total ticks: {df_new['tick'].max()}")
    print(f"📊 Total actions: {len(df_new)}")
    
    print("\n📊 Action distribution:")
    actions = df_new['action'].value_counts()
    for action, count in actions.items():
        pct = count / len(df_new) * 100
        print(f"  {action:<15} {count:>5} ({pct:>5.1f}%)")
    
    eat_actions = df_new[df_new['action'] == 'EAT']
    if len(eat_actions) > 0:
        eat_success = eat_actions['success'].sum() / len(eat_actions) * 100
        print(f"\n🍎 EAT success rate: {eat_success:.1f}%")
        print(f"🍎 Successful EATs: {eat_actions['success'].sum()}")

print("\n" + "=" * 70)
