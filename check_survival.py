#!/usr/bin/env python3
"""Quick check of agent survival from most recent test."""

import pandas as pd

# Load the most recent agent states log
df = pd.read_csv('data/logs/agent_states_20251116_020241.csv')

print("=" * 60)
print("SURVIVAL ANALYSIS - Test from 2025-11-16 02:02:41")
print("=" * 60)

# Find when each agent died (first tick where alive=False)
all_agents = df['agent_id'].unique()
print(f"\nTotal agents: {len(all_agents)}")

death_info = []
for agent_id in all_agents:
    agent_data = df[df['agent_id'] == agent_id]
    death_tick = agent_data[agent_data['alive'] == False]['tick'].min()
    
    if pd.isna(death_tick):
        # Agent never died
        last_tick = agent_data['tick'].max()
        last_age = agent_data[agent_data['tick'] == last_tick]['age'].iloc[0]
        death_info.append({
            'agent_id': agent_id,
            'death_tick': 'ALIVE',
            'age_at_death': f'{int(last_age)} (still alive)',
            'survival': 'SURVIVED'
        })
    else:
        death_age = agent_data[agent_data['tick'] == death_tick]['age'].iloc[0]
        death_info.append({
            'agent_id': agent_id,
            'death_tick': int(death_tick),
            'age_at_death': int(death_age),
            'survival': 'DIED'
        })

# Sort by death tick (put ALIVE at end)
death_df = pd.DataFrame(death_info)
death_df_sorted = death_df.sort_values(
    by=['survival', 'death_tick'],
    ascending=[True, True],
    key=lambda x: x.map({'DIED': 0, 'ALIVE': 1}) if x.name == 'survival' else x
)

print("\nAgent Survival Summary:")
print(death_df_sorted.to_string(index=False))

# Calculate statistics
died_agents = death_df[death_df['survival'] == 'DIED']
alive_agents = death_df[death_df['survival'] == 'SURVIVED']

print("\n" + "=" * 60)
print("STATISTICS")
print("=" * 60)
print(f"Agents that died: {len(died_agents)}")
print(f"Agents still alive at tick 1000: {len(alive_agents)}")

if len(died_agents) > 0:
    avg_death_age = died_agents['age_at_death'].mean()
    print(f"\nAverage death age (for those who died): {avg_death_age:.1f} ticks")
    print(f"Earliest death: {died_agents['age_at_death'].min()} ticks")
    print(f"Latest death: {died_agents['age_at_death'].max()} ticks")

if len(alive_agents) > 0:
    print(f"\n✅ {len(alive_agents)} agents reached tick 1000 ALIVE!")
    for _, agent in alive_agents.iterrows():
        print(f"   Agent {agent['agent_id']}: {agent['age_at_death']}")

# Check final state at tick 1000
final_state = df[df['tick'] == 1000]
print(f"\n" + "=" * 60)
print(f"FINAL STATE AT TICK 1000")
print("=" * 60)
print(f"Agents alive: {len(final_state)}")
if len(final_state) > 0:
    print(f"Average energy: {final_state['energy'].mean():.1f}")
    print(f"Average age: {final_state['age'].mean():.1f}")
    
# Check tick distribution
print(f"\n" + "=" * 60)
print(f"TICK DISTRIBUTION")
print("=" * 60)
tick_counts = df.groupby('tick').size()
print(f"First tick: {tick_counts.index.min()}")
print(f"Last tick: {tick_counts.index.max()}")
print(f"Total ticks logged: {len(tick_counts)}")
print(f"\nAgents per tick (last 10 ticks):")
for tick in sorted(tick_counts.index)[-10:]:
    count = tick_counts[tick]
    print(f"  Tick {tick}: {count} agents")
