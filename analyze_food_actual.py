"""
Actual Food Calories Analysis

Parses the message field to get the ACTUAL calories from food,
not the net change which is affected by max_energy cap.
"""

import pandas as pd
import re
import os


def find_latest_log():
    """Find the most recent log file."""
    log_dir = "data/logs"
    action_files = [f for f in os.listdir(log_dir) if f.startswith("agent_actions_")]
    
    if not action_files:
        return None
    
    latest_action = sorted(action_files)[-1]
    return os.path.join(log_dir, latest_action)


def parse_food_calories():
    """Parse actual food calories from eat action messages."""
    action_file = find_latest_log()
    
    if not action_file:
        print("❌ No log files found!")
        return
    
    print("=" * 80)
    print("🍎 ACTUAL FOOD CALORIES ANALYSIS")
    print("=" * 80)
    print(f"\n📄 Analyzing: {os.path.basename(action_file)}\n")
    
    # Load data
    actions = pd.read_csv(action_file)
    
    # Get successful eats
    successful_eats = actions[
        (actions['action'] == 'EAT') & 
        (actions['success'] == True)
    ].copy()
    
    if len(successful_eats) == 0:
        print("❌ No successful eat actions found!")
        return
    
    # Parse calories from message
    # Message format: "Ate food, gained X.X energy"
    def extract_calories(message):
        match = re.search(r'gained ([\d.]+) energy', message)
        if match:
            return float(match.group(1))
        return 0.0
    
    successful_eats['actual_calories'] = successful_eats['message'].apply(extract_calories)
    
    # Get statistics
    total_calories = successful_eats['actual_calories'].sum()
    num_eats = len(successful_eats)
    avg_calories = total_calories / num_eats if num_eats > 0 else 0
    min_calories = successful_eats['actual_calories'].min()
    max_calories = successful_eats['actual_calories'].max()
    
    print(f"📊 SUMMARY:")
    print(f"  Total successful eats: {num_eats}")
    print(f"  Total calories gained: {total_calories:.1f}")
    print(f"  Average per eat: {avg_calories:.1f}")
    print(f"  Min calories: {min_calories:.1f}")
    print(f"  Max calories: {max_calories:.1f}")
    
    # Show histogram
    print(f"\n📈 CALORIE DISTRIBUTION:")
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    hist = pd.cut(successful_eats['actual_calories'], bins=bins).value_counts().sort_index()
    
    for interval, count in hist.items():
        if count > 0:
            bar = "█" * int(count)
            print(f"  {interval}: {count:2d} {bar}")
    
    # Check for freshness decay
    print(f"\n🥗 FRESHNESS ANALYSIS:")
    if max_calories > 0:
        freshness_avg = avg_calories / 35.0  # Assuming 35 base calories
        print(f"  Average freshness: {freshness_avg:.2f} ({freshness_avg*100:.1f}%)")
        print(f"  Expected (fresh): 1.00 (100%)")
        
        if freshness_avg < 0.95:
            print(f"  ⚠️  Significant freshness decay detected!")
        elif freshness_avg < 0.80:
            print(f"  🚨 SEVERE freshness decay!")
        else:
            print(f"  ✅ Freshness looks good")
    
    # Per-agent analysis
    print(f"\n📋 PER-AGENT CALORIES:")
    print(f"{'Agent':<10} {'Eats':<8} {'Total Cal':<12} {'Avg Cal':<10} {'Freshness':<12}")
    print("-" * 60)
    
    for agent_id in sorted(successful_eats['agent_id'].unique()):
        agent_eats = successful_eats[successful_eats['agent_id'] == agent_id]
        agent_total = agent_eats['actual_calories'].sum()
        agent_avg = agent_eats['actual_calories'].mean()
        agent_freshness = agent_avg / 35.0 if agent_avg > 0 else 0
        
        print(f"{agent_id:<10.0f} {len(agent_eats):<8} {agent_total:<12.1f} {agent_avg:<10.1f} {agent_freshness:<12.2f}")
    
    # Timeline analysis - check if freshness decays over time
    print(f"\n⏱️  TIMELINE ANALYSIS:")
    successful_eats['time_bin'] = pd.cut(successful_eats['tick'], bins=5)
    timeline = successful_eats.groupby('time_bin')['actual_calories'].agg(['count', 'mean'])
    
    print(f"{'Time Range':<25} {'Eats':<8} {'Avg Calories':<15}")
    print("-" * 50)
    for idx, row in timeline.iterrows():
        if row['count'] > 0:
            print(f"{str(idx):<25} {row['count']:<8.0f} {row['mean']:<15.1f}")
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    # Return summary for further use
    return {
        'total_eats': num_eats,
        'total_calories': total_calories,
        'avg_calories': avg_calories,
        'avg_freshness': avg_calories / 35.0 if avg_calories > 0 else 0
    }


if __name__ == "__main__":
    results = parse_food_calories()
    
    if results:
        print(f"\n💡 KEY FINDING:")
        expected_calories = 35.0
        actual_avg = results['avg_calories']
        
        if actual_avg < expected_calories * 0.5:
            print(f"  🚨 CRITICAL: Food calories are {actual_avg:.1f}, expected {expected_calories:.1f}")
            print(f"     This is only {(actual_avg/expected_calories*100):.1f}% of expected!")
            print(f"     Likely cause: Severe freshness decay")
        elif actual_avg < expected_calories * 0.9:
            print(f"  ⚠️  WARNING: Food calories are {actual_avg:.1f}, expected {expected_calories:.1f}")
            print(f"     This is {(actual_avg/expected_calories*100):.1f}% of expected")
            print(f"     Likely cause: Freshness decay over time")
        else:
            print(f"  ✅ Food calories look correct: {actual_avg:.1f} (expected {expected_calories:.1f})")
