"""Analyze latest action/transition logs with compatibility for old and new schemas."""

import glob
import os
import sys
import pandas as pd
import numpy as np


def _pick_latest_log(log_dir: str):
    action_files = glob.glob(os.path.join(log_dir, 'agent_actions_*.csv'))
    transition_files = glob.glob(os.path.join(log_dir, 'transitions_*.csv'))
    all_files = action_files + transition_files
    if not all_files:
        return None, False
    latest = max(all_files, key=os.path.getctime)
    return latest, ('transitions_' in os.path.basename(latest))


def _coerce_bool_success(df: pd.DataFrame) -> None:
    if 'success' in df.columns:
        if df['success'].dtype == bool:
            return
        df['success'] = df['success'].astype(str).str.lower().isin({'1', 'true', 'yes'})


def _ensure_interaction_fields(df: pd.DataFrame) -> None:
    if 'result_message' not in df.columns and 'message' in df.columns:
        df['result_message'] = df['message'].fillna('')
    elif 'result_message' not in df.columns:
        df['result_message'] = ''

    if 'interaction_kind' not in df.columns:
        df['interaction_kind'] = ''
    if 'object_type' not in df.columns:
        df['object_type'] = ''
    if 'object_id' not in df.columns:
        df['object_id'] = -1
    if 'target_x' not in df.columns:
        df['target_x'] = -1
    if 'target_y' not in df.columns:
        df['target_y'] = -1

    # Backfill from message for older logs
    message = df['result_message'].fillna('').astype(str)

    missing_obj_id = pd.to_numeric(df['object_id'], errors='coerce').fillna(-1) < 0
    extracted_id = message.str.extract(r'(?:\bfood\b|\bseed\b|\bfertilizer\b|\bobject\b)\s+(\d+)')[0]
    extracted_id = pd.to_numeric(extracted_id, errors='coerce').fillna(-1).astype(int)
    df.loc[missing_obj_id, 'object_id'] = extracted_id[missing_obj_id]

    missing_obj_type = df['object_type'].fillna('').astype(str).str.len() == 0
    inferred_type = pd.Series([''] * len(df), index=df.index)
    inferred_type[message.str.contains(r'\bseed\b', case=False)] = 'seed'
    inferred_type[message.str.contains(r'\bfertilizer\b', case=False)] = 'fertilizer'
    inferred_type[message.str.contains(r'\bfood\b', case=False)] = 'food'
    inferred_type[message.str.contains(r'\bplant\b', case=False)] = 'plant'
    inferred_type[message.str.contains(r'\bobject\b', case=False) & (inferred_type == '')] = 'object'
    df.loc[missing_obj_type, 'object_type'] = inferred_type[missing_obj_type]

    missing_kind = df['interaction_kind'].fillna('').astype(str).str.len() == 0
    inferred_kind = pd.Series([''] * len(df), index=df.index)
    inferred_kind[df['action'] == 'PICK_UP'] = 'pickup'
    inferred_kind[df['action'] == 'DROP'] = 'drop_here'
    inferred_kind[df['action'] == 'USE'] = 'use'
    inferred_kind[df['action'] == 'EAT'] = 'eat'
    inferred_kind[(df['action'] == 'DROP') & message.str.contains('nearby', case=False)] = 'drop_nearby'
    inferred_kind[(df['action'] == 'USE') & message.str.contains('planted seed', case=False)] = 'plant_seed'
    inferred_kind[(df['action'] == 'USE') & message.str.contains('applied fertilizer', case=False)] = 'apply_fertilizer'
    df.loc[missing_kind, 'interaction_kind'] = inferred_kind[missing_kind]

    # Parse target coords when present in messages like "(x, y)"
    parsed_xy = message.str.extract(r'\(([-]?\d+)\s*,\s*([-]?\d+)\)')
    parsed_x = pd.to_numeric(parsed_xy[0], errors='coerce')
    parsed_y = pd.to_numeric(parsed_xy[1], errors='coerce')
    target_x = pd.to_numeric(df['target_x'], errors='coerce').fillna(-1)
    target_y = pd.to_numeric(df['target_y'], errors='coerce').fillna(-1)
    df.loc[target_x < 0, 'target_x'] = parsed_x[target_x < 0].fillna(-1).astype(int)
    df.loc[target_y < 0, 'target_y'] = parsed_y[target_y < 0].fillna(-1).astype(int)


def _derive_position_columns(df: pd.DataFrame) -> tuple[str, str] | tuple[None, None]:
    if 'x' in df.columns and 'y' in df.columns:
        return 'x', 'y'
    if 'x_after' in df.columns and 'y_after' in df.columns:
        return 'x_after', 'y_after'
    if 'x_before' in df.columns and 'y_before' in df.columns:
        return 'x_before', 'y_before'
    return None, None


def _compute_loop_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    """Compute lightweight path-loop indicators from action logs."""
    has_before_after = {'x_before', 'y_before', 'x_after', 'y_after'}.issubset(df.columns)
    x_col, y_col = _derive_position_columns(df)
    if not has_before_after and x_col is None:
        return {}

    work = df.copy()
    work = work.sort_values(['agent_id', 'tick'])

    backtracks = 0
    abab_cycles = 0
    reactive_turns = 0
    blocked_forwards = 0
    proactive_turns_after_run = 0
    total_turns = 0
    long_straight_runs = 0
    max_straight_run = 0
    return_cycles_3_10 = 0
    revisit_within_8 = 0
    moved_steps = 0

    for _, group in work.groupby('agent_id', sort=False):
        if has_before_after:
            positions = list(zip(group['x_before'].tolist(), group['y_before'].tolist()))
            if len(group) > 0:
                positions.append((int(group.iloc[-1]['x_after']), int(group.iloc[-1]['y_after'])))
        else:
            positions = list(zip(group[x_col].tolist(), group[y_col].tolist()))

        if not positions:
            continue

        # Movement-only path: collapse consecutive duplicates so cycle metrics
        # reflect spatial loops, not prolonged WAIT/idling on one tile.
        move_positions = [positions[0]]
        for pos in positions[1:]:
            if pos != move_positions[-1]:
                move_positions.append(pos)

        actions = group['action'].tolist() if 'action' in group.columns else []
        success = group['success'].tolist() if 'success' in group.columns else [True] * len(group)

        # Position cycle indicators
        for i in range(2, len(move_positions)):
            if move_positions[i] == move_positions[i - 2] and move_positions[i] != move_positions[i - 1]:
                backtracks += 1

        for i in range(3, len(move_positions)):
            if (
                move_positions[i] == move_positions[i - 2]
                and move_positions[i - 1] == move_positions[i - 3]
                and move_positions[i] != move_positions[i - 1]
            ):
                abab_cycles += 1

        # Generic short return cycles (length 3..10): position[t] == position[t-k]
        # Captures broader loop motifs (e.g., ABCAB-like and longer patrol loops).
        moved_steps += max(0, len(move_positions) - 1)
        for i in range(1, len(move_positions)):
            found_recent_revisit = False
            for k in range(3, 11):
                if i - k < 0:
                    continue
                if move_positions[i] == move_positions[i - k]:
                    return_cycles_3_10 += 1
                    found_recent_revisit = True
                    break

            if found_recent_revisit:
                revisit_within_8 += 1

        # Action-pattern indicators
        forward_run = 0
        blocked_recent = 0
        for act, ok in zip(actions, success):
            if act == 'MOVE_FORWARD':
                if ok:
                    forward_run += 1
                else:
                    blocked_forwards += 1
                    blocked_recent = 2
                    forward_run = 0
            else:
                if act in ('TURN_LEFT', 'TURN_RIGHT'):
                    total_turns += 1
                    if blocked_recent > 0:
                        reactive_turns += 1
                    elif forward_run >= 3:
                        proactive_turns_after_run += 1

                if forward_run >= 6:
                    long_straight_runs += 1
                max_straight_run = max(max_straight_run, forward_run)
                forward_run = 0

                if blocked_recent > 0:
                    blocked_recent -= 1

        if forward_run >= 6:
            long_straight_runs += 1
        max_straight_run = max(max_straight_run, forward_run)

    total_actions = max(len(work), 1)
    moved_steps = max(moved_steps, 1)
    return {
        'backtrack_events': int(backtracks),
        'abab_cycles': int(abab_cycles),
        'return_cycles_3_10': int(return_cycles_3_10),
        'revisit_within_8': int(revisit_within_8),
        'blocked_forwards': int(blocked_forwards),
        'reactive_turns': int(reactive_turns),
        'proactive_turns_after_run': int(proactive_turns_after_run),
        'total_turns': int(total_turns),
        'long_straight_runs': int(long_straight_runs),
        'max_straight_run': int(max_straight_run),
        'backtrack_rate_pct': (backtracks / total_actions) * 100.0,
        'abab_rate_pct': (abab_cycles / total_actions) * 100.0,
        'return_cycles_3_10_pct': (return_cycles_3_10 / total_actions) * 100.0,
        'revisit_within_8_move_pct': (revisit_within_8 / moved_steps) * 100.0,
        'reactive_turn_share_pct': ((reactive_turns / total_turns) * 100.0) if total_turns > 0 else 0.0,
    }


# ── NEW METRIC FUNCTIONS ────────────────────────────────────────────


def _compute_energy_metrics(df: pd.DataFrame) -> dict:
    """Energy economy: gains, losses, efficiency, net per tick."""
    if 'energy_before' not in df.columns or 'energy_after' not in df.columns:
        return {}
    delta = df['energy_after'] - df['energy_before']
    gains = delta[delta > 0]
    losses = delta[delta < 0]
    # Per-action energy cost
    cost_by_action = df.groupby('action')['energy_cost'].mean()
    return {
        'avg_energy_delta': float(delta.mean()),
        'total_energy_gained': float(gains.sum()),
        'total_energy_lost': float(losses.sum()),
        'avg_energy_per_tick': float(delta.sum() / max(df['tick'].nunique(), 1)),
        'cost_by_action': cost_by_action.to_dict(),
    }


def _compute_population_dynamics(df: pd.DataFrame) -> dict:
    """Population over time: peak, extinction risk, growth rate."""
    agents_per_tick = df.groupby('tick')['agent_id'].nunique()
    peak_pop = int(agents_per_tick.max())
    min_pop = int(agents_per_tick.min())
    mean_pop = float(agents_per_tick.mean())

    # First & last agent IDs to count births / deaths
    all_agents = df['agent_id'].unique()
    first_seen = df.groupby('agent_id')['tick'].min()
    last_seen = df.groupby('agent_id')['tick'].max()
    max_tick = int(df['tick'].max())

    # Agents whose last seen tick < max_tick likely died
    deaths = int((last_seen < max_tick).sum())
    # Agents whose first seen tick > 0 were born during the sim
    births = int((first_seen > 0).sum())

    return {
        'total_unique_agents': int(len(all_agents)),
        'peak_population': peak_pop,
        'min_population': min_pop,
        'mean_population': round(mean_pop, 1),
        'estimated_births': births,
        'estimated_deaths': deaths,
        'pop_by_tick': agents_per_tick,  # Series for optional plotting
    }


def _compute_lifespan_metrics(df: pd.DataFrame) -> dict:
    """Per-agent lifespan and lifetime stats."""
    agent_stats = df.groupby('agent_id').agg(
        first_tick=('tick', 'min'),
        last_tick=('tick', 'max'),
        num_actions=('tick', 'count'),
        max_age=('age', 'max'),
        peak_fitness=('fitness', 'max'),
        max_energy=('energy_before', 'max'),
        min_energy=('energy_before', 'min'),
        total_eats=('action', lambda x: (x == 'EAT').sum()),
        total_pickups=('action', lambda x: (x == 'PICK_UP').sum()),
    )
    agent_stats['lifespan'] = agent_stats['last_tick'] - agent_stats['first_tick']

    return {
        'avg_lifespan': float(agent_stats['lifespan'].mean()),
        'median_lifespan': float(agent_stats['lifespan'].median()),
        'max_lifespan': int(agent_stats['lifespan'].max()),
        'min_lifespan': int(agent_stats['lifespan'].min()),
        'std_lifespan': float(agent_stats['lifespan'].std()),
        'avg_peak_fitness': float(agent_stats['peak_fitness'].mean()),
        'max_peak_fitness': float(agent_stats['peak_fitness'].max()),
        'avg_eats_per_agent': float(agent_stats['total_eats'].mean()),
        'avg_pickups_per_agent': float(agent_stats['total_pickups'].mean()),
        'top5_agents': agent_stats.nlargest(5, 'peak_fitness')[
            ['lifespan', 'peak_fitness', 'total_eats', 'total_pickups', 'max_age']
        ],
    }


def _compute_spatial_metrics(df: pd.DataFrame) -> dict:
    """Spatial coverage, territory, displacement."""
    x_col, y_col = _derive_position_columns(df)
    if x_col is None:
        return {}

    # Per-agent unique tiles visited
    per_agent = df.groupby('agent_id').apply(
        lambda g: len(set(zip(g[x_col].tolist(), g[y_col].tolist()))),
        include_groups=False,
    )
    # Global heatmap
    all_positions = list(zip(df[x_col].tolist(), df[y_col].tolist()))
    unique_tiles_global = len(set(all_positions))

    # Displacement: distance from first to last position per agent
    displacements = []
    for _, g in df.groupby('agent_id'):
        x0, y0 = g.iloc[0][x_col], g.iloc[0][y_col]
        x1, y1 = g.iloc[-1][x_col], g.iloc[-1][y_col]
        displacements.append(((x1 - x0)**2 + (y1 - y0)**2)**0.5)
    displacements = np.array(displacements)

    # Hotspot: most visited tile
    from collections import Counter
    tile_counts = Counter(all_positions)
    hotspot_tile, hotspot_visits = tile_counts.most_common(1)[0]

    return {
        'unique_tiles_global': unique_tiles_global,
        'avg_tiles_per_agent': float(per_agent.mean()),
        'median_tiles_per_agent': float(per_agent.median()),
        'max_tiles_per_agent': int(per_agent.max()),
        'avg_displacement': float(displacements.mean()),
        'max_displacement': float(displacements.max()),
        'hotspot_tile': hotspot_tile,
        'hotspot_visits': int(hotspot_visits),
    }


def _compute_temporal_phases(df: pd.DataFrame, n_phases: int = 4) -> dict:
    """Split simulation into N equal phases and compare action distributions."""
    max_tick = int(df['tick'].max())
    if max_tick == 0:
        return {}
    phase_len = max(1, max_tick // n_phases)
    df = df.copy()
    df['phase'] = (df['tick'] // phase_len).clip(upper=n_phases - 1)

    phase_data = {}
    for phase in range(n_phases):
        p = df[df['phase'] == phase]
        total = max(len(p), 1)
        dist = p['action'].value_counts(normalize=True).to_dict()
        n_agents = int(p['agent_id'].nunique())
        avg_energy = float(p['energy_before'].mean()) if 'energy_before' in p.columns else 0.0
        ticks = f"{phase * phase_len}-{min((phase + 1) * phase_len, max_tick)}"
        phase_data[phase] = {
            'ticks': ticks,
            'n_agents': n_agents,
            'n_actions': len(p),
            'avg_energy': round(avg_energy, 1),
            'action_dist': {k: round(v * 100, 1) for k, v in dist.items()},
        }
    return phase_data


def _compute_farming_metrics(df: pd.DataFrame) -> dict:
    """Seed planting pipeline: pickup → plant → harvest chain."""
    seed_pickups = df[(df['action'] == 'PICK_UP') & (df['object_type'] == 'seed')]
    seed_plants = df[(df['action'] == 'USE') & (df['interaction_kind'].str.contains('plant_seed', na=False))]
    food_pickups = df[(df['action'] == 'PICK_UP') & (df['object_type'] == 'food')]
    eats = df[df['action'] == 'EAT']

    # Agents who planted at least once
    planters = seed_plants['agent_id'].unique()
    # Agents who picked up seeds
    seed_collectors = seed_pickups['agent_id'].unique()

    return {
        'total_seed_pickups': len(seed_pickups),
        'total_seed_plants': len(seed_plants),
        'total_food_pickups': len(food_pickups),
        'total_eats': len(eats),
        'unique_planters': len(planters),
        'unique_seed_collectors': len(seed_collectors),
        'plant_to_pickup_ratio': round(len(seed_plants) / max(len(seed_pickups), 1), 3),
        'food_per_eat': round(len(food_pickups) / max(len(eats), 1), 2),
    }


def _compute_behavioral_diversity(df: pd.DataFrame) -> dict:
    """How diverse are individual agent strategies?"""
    all_actions = sorted(df['action'].unique())
    per_agent_dist = df.groupby('agent_id')['action'].value_counts(normalize=True).unstack(fill_value=0)
    # Ensure all action columns exist
    for a in all_actions:
        if a not in per_agent_dist.columns:
            per_agent_dist[a] = 0.0
    per_agent_dist = per_agent_dist[all_actions]

    # Shannon entropy per agent
    def entropy(row):
        p = row[row > 0]
        return float(-np.sum(p * np.log2(p)))

    entropies = per_agent_dist.apply(entropy, axis=1)
    max_entropy = np.log2(len(all_actions))

    # Std of action proportions across agents (high = diverse strategies)
    action_std = per_agent_dist.std()

    return {
        'avg_entropy': float(entropies.mean()),
        'max_possible_entropy': float(max_entropy),
        'entropy_ratio': float(entropies.mean() / max_entropy) if max_entropy > 0 else 0,
        'min_entropy_agent': int(entropies.idxmin()),
        'max_entropy_agent': int(entropies.idxmax()),
        'action_std_across_agents': action_std.to_dict(),
    }


def _compute_efficiency_metrics(df: pd.DataFrame) -> dict:
    """Energy efficiency: calories gained per energy spent."""
    # Total energy spent (sum of energy_cost)
    total_cost = float(df['energy_cost'].sum()) if 'energy_cost' in df.columns else 0.0

    # Total energy gained from eating
    eat_rows = df[(df['action'] == 'EAT') & (df['success'] == True)]
    total_calories = 0.0
    if 'energy_before' in df.columns and 'energy_after' in df.columns and len(eat_rows) > 0:
        eat_gain = eat_rows['energy_after'] - eat_rows['energy_before'] + eat_rows['energy_cost']
        total_calories = float(eat_gain[eat_gain > 0].sum())

    # Movement efficiency: unique tiles visited per MOVE action
    move_rows = df[(df['action'] == 'MOVE_FORWARD') & (df['success'] == True)]
    x_col, y_col = _derive_position_columns(df)
    unique_tiles_moved = 0
    if x_col and len(move_rows) > 0:
        unique_tiles_moved = len(set(zip(move_rows[x_col].tolist(), move_rows[y_col].tolist())))

    return {
        'total_energy_spent': round(total_cost, 1),
        'total_calories_eaten': round(total_calories, 1),
        'caloric_efficiency': round(total_calories / max(total_cost, 1), 3),
        'total_moves': len(move_rows),
        'unique_tiles_via_move': unique_tiles_moved,
        'move_exploration_ratio': round(unique_tiles_moved / max(len(move_rows), 1), 3),
    }


def _compute_action_sequences(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Most common 2-gram and 3-gram action sequences."""
    from collections import Counter
    bigrams: Counter = Counter()
    trigrams: Counter = Counter()
    for _, g in df.groupby('agent_id', sort=False):
        actions = g['action'].tolist()
        for i in range(len(actions) - 1):
            bigrams[(actions[i], actions[i + 1])] += 1
        for i in range(len(actions) - 2):
            trigrams[(actions[i], actions[i + 1], actions[i + 2])] += 1
    return {
        'top_bigrams': [(f"{a}→{b}", c) for (a, b), c in bigrams.most_common(top_n)],
        'top_trigrams': [(f"{a}→{b}→{c}", n) for (a, b, c), n in trigrams.most_common(top_n)],
    }


log_dir = 'data/logs'
latest_file, is_new_format = _pick_latest_log(log_dir)

if latest_file is None:
    print("❌ No log files found!")
    print(f"  Searched in: {log_dir}")
    print("  Looking for: agent_actions_*.csv or transitions_*.csv")
    raise SystemExit(1)

print(f"📊 Analyzing: {latest_file}")
print(f"   Format: {'WorldModelLogger (new)' if is_new_format else 'AgentLogger (old)'}")
print("=" * 70)

df = pd.read_csv(latest_file, low_memory=False)
_coerce_bool_success(df)
if 'death_reason' in df.columns:
    df['death_reason'] = df['death_reason'].fillna('')
_ensure_interaction_fields(df)

total_actions = len(df)
total_ticks = int(df['tick'].max()) if total_actions > 0 else 0
num_agents = int(df['agent_id'].nunique()) if total_actions > 0 else 0

print("\n📈 SIMULATION SUMMARY")
print(f"  Total ticks: {total_ticks}")
print(f"  Total actions: {total_actions:,}")
print(f"  Number of agents: {num_agents}")

if is_new_format:
    episodes_file = latest_file.replace('transitions_', 'episodes_')
    world_states_file = latest_file.replace('transitions_', 'world_states_')

    if os.path.exists(episodes_file):
        episodes_df = pd.read_csv(episodes_file)
        if len(episodes_df) > 0:
            print("\n📋 EPISODE SUMMARY")
            print(f"  Total episodes: {len(episodes_df)}")
            print(f"  Avg duration: {episodes_df['duration'].mean():.1f} ticks")
            print(f"  Avg reward: {episodes_df['total_reward'].mean():.2f}")
            print(f"  Avg successful eats: {episodes_df['successful_eats'].mean():.1f}")
            print(f"  Avg tiles explored: {episodes_df['tiles_explored'].mean():.1f}")

    if os.path.exists(world_states_file):
        world_df = pd.read_csv(world_states_file)
        if len(world_df) > 0:
            print("\n🌍 WORLD STATE SUMMARY")
            print(f"  Avg food count: {world_df['total_food'].mean():.1f}")
            print(f"  Avg plants: {world_df['total_plants'].mean():.1f}")
            print(f"  Avg agent energy: {world_df['avg_agent_energy'].mean():.1f}")
            print(f"  Max agent age reached: {world_df['max_agent_age'].max()}")

if 'reward' in df.columns:
    print("\n💰 REWARD ANALYSIS")
    print(f"  Total reward: {df['reward'].sum():.2f}")
    print(f"  Avg reward per action: {df['reward'].mean():.4f}")
    print(f"  Max reward: {df['reward'].max():.2f}")
    print(f"  Min reward: {df['reward'].min():.2f}")
    print("\n  Reward by action:")
    reward_by_action = df.groupby('action')['reward'].agg(['mean', 'sum', 'count'])
    for action, row in reward_by_action.iterrows():
        print(f"    {action:15s}: avg={row['mean']:+.3f}, total={row['sum']:+.1f}, count={int(row['count'])}")

print("\n🎯 ACTION DISTRIBUTION")
action_counts = df['action'].value_counts()
for action, count in action_counts.items():
    pct = (count / max(total_actions, 1)) * 100
    print(f"  {action:15s}: {count:5d} ({pct:5.1f}%)")

print("\n✅ SUCCESS RATES")
for action in df['action'].unique():
    action_df = df[df['action'] == action]
    success_rate = (action_df['success'].sum() / max(len(action_df), 1)) * 100
    print(f"  {action:15s}: {success_rate:5.1f}%")

loop_metrics = _compute_loop_metrics(df)
if loop_metrics:
    print("\n🔁 PATH-LOOP METRICS")
    print(f"  Backtrack events (A→B→A): {loop_metrics['backtrack_events']:,} ({loop_metrics['backtrack_rate_pct']:.2f}%)")
    print(f"  ABAB short cycles:         {loop_metrics['abab_cycles']:,} ({loop_metrics['abab_rate_pct']:.2f}%)")
    print(f"  Return cycles (len 3-10):  {loop_metrics['return_cycles_3_10']:,} ({loop_metrics['return_cycles_3_10_pct']:.2f}%)")
    print(f"  Moved-step revisits ≤8:    {loop_metrics['revisit_within_8']:,} ({loop_metrics['revisit_within_8_move_pct']:.1f}% of moved steps)")
    print(f"  Blocked MOVE_FORWARD:      {loop_metrics['blocked_forwards']:,}")
    print(f"  Total turns:               {loop_metrics['total_turns']:,}")
    print(f"  Reactive turns:            {loop_metrics['reactive_turns']:,} ({loop_metrics['reactive_turn_share_pct']:.1f}% of turns)")
    print(f"  Proactive turns after run: {loop_metrics['proactive_turns_after_run']:,}")
    print(f"  Long straight runs (>=6):  {loop_metrics['long_straight_runs']:,}")
    print(f"  Max straight run length:   {loop_metrics['max_straight_run']}")
else:
    print("\n🔁 PATH-LOOP METRICS")
    print("  Skipped (position columns not found in this log schema)")

print("\n🧰 INTERACTION DETAIL (PICK_UP / DROP / USE)")
for action_name in ['PICK_UP', 'DROP', 'USE']:
    subset = df[df['action'] == action_name]
    if len(subset) == 0:
        print(f"  {action_name:8s}: no events")
        continue

    success_subset = subset[subset['success'] == True]
    fail_subset = subset[subset['success'] == False]
    print(f"  {action_name:8s}: total={len(subset)}, success={len(success_subset)}, fail={len(fail_subset)}")

    if len(success_subset) > 0:
        by_kind = success_subset['interaction_kind'].fillna('').replace('', 'unknown').value_counts()
        by_type = success_subset['object_type'].fillna('').replace('', 'unknown').value_counts()
        print("    success kinds:")
        for kind, count in by_kind.head(6).items():
            print(f"      - {kind}: {count}")
        print("    success object types:")
        for obj_type, count in by_type.head(6).items():
            print(f"      - {obj_type}: {count}")

    if len(fail_subset) > 0:
        reason_counts = fail_subset['result_message'].fillna('').replace('', '<no_message>').value_counts()
        print("    top failure reasons:")
        for reason, count in reason_counts.head(5).items():
            print(f"      - {reason}: {count}")

    examples = success_subset.head(5)
    if len(examples) > 0:
        print("    sample successful events:")
        for _, row in examples.iterrows():
            obj = row['object_type'] if str(row['object_type']) else 'unknown'
            oid = int(row['object_id']) if pd.notna(row['object_id']) else -1
            tx = int(row['target_x']) if pd.notna(row['target_x']) else -1
            ty = int(row['target_y']) if pd.notna(row['target_y']) else -1
            print(
                f"      - tick={int(row['tick'])}, agent={int(row['agent_id'])}, "
                f"kind={row['interaction_kind']}, obj={obj}#{oid}, target=({tx},{ty})"
            )

eat_actions = df[df['action'] == 'EAT']
eat_success_rate = 0.0
if len(eat_actions) > 0:
    eat_success = int(eat_actions['success'].sum())
    eat_attempts = len(eat_actions)
    eat_success_rate = (eat_success / max(eat_attempts, 1)) * 100
    print("\n🍎 FOOD CONSUMPTION")
    print(f"  EAT attempts: {eat_attempts}")
    print(f"  Successful: {eat_success}")
    print(f"  Success rate: {eat_success_rate:.1f}%")

move_actions = df[df['action'] == 'MOVE_FORWARD']
if len(move_actions) > 0:
    move_success = int(move_actions['success'].sum())
    move_attempts = len(move_actions)
    move_success_rate = (move_success / max(move_attempts, 1)) * 100
    print("\n🚶 MOVEMENT")
    print(f"  MOVE attempts: {move_attempts}")
    print(f"  Successful: {move_success}")
    print(f"  Success rate: {move_success_rate:.1f}%")

wait_actions = df[df['action'] == 'WAIT']
wait_count = len(wait_actions)
wait_pct = (wait_count / max(total_actions, 1)) * 100
print("\n⏸️  WAITING BEHAVIOR")
print(f"  WAIT actions: {wait_count}")
print(f"  Percentage: {wait_pct:.1f}%")

print("\n🎯 V3.1 TARGET COMPARISON")
print("  WAIT actions:")
print(f"    Actual: {wait_pct:.1f}%")
print("    Target: 32-35%")
if 32 <= wait_pct <= 35:
    print("    Status: ✅ WITHIN TARGET")
elif wait_pct < 32:
    print("    Status: ⚠️  TOO AGGRESSIVE (decrease exploration bonus)")
else:
    print("    Status: ⚠️  TOO PASSIVE (increase exploration bonus)")

if len(eat_actions) > 0:
    print("  EAT success rate:")
    print(f"    Actual: {eat_success_rate:.1f}%")
    print("    Target: 4-5%")
    if 4 <= eat_success_rate <= 5:
        print("    Status: ✅ WITHIN TARGET")
    elif eat_success_rate < 3:
        print("    Status: ⚠️  TOO LOW (agents not finding/eating food)")
    else:
        print("    Status: ✅ ABOVE TARGET (good!)")

print("  Survival time:")
print(f"    Actual: {total_ticks} ticks")
print("    Target: >1500 ticks")
if total_ticks >= 1500:
    print("    Status: ✅ TARGET MET")
else:
    print("    Status: ⚠️  BELOW TARGET")

# ── NEW: ENERGY ECONOMY ─────────────────────────────────────────────

energy_m = _compute_energy_metrics(df)
if energy_m:
    print("\n⚡ ENERGY ECONOMY")
    print(f"  Avg energy change per action: {energy_m['avg_energy_delta']:+.3f}")
    print(f"  Total energy gained (eating): {energy_m['total_energy_gained']:,.1f}")
    print(f"  Total energy lost (costs):    {energy_m['total_energy_lost']:,.1f}")
    print(f"  Net energy per tick:          {energy_m['avg_energy_per_tick']:+.2f}")
    print("  Avg cost per action type:")
    for act, cost in sorted(energy_m['cost_by_action'].items(), key=lambda x: -x[1]):
        print(f"    {act:15s}: {cost:.3f}")

# ── NEW: CALORIC EFFICIENCY ──────────────────────────────────────────

eff_m = _compute_efficiency_metrics(df)
if eff_m:
    print("\n📊 CALORIC EFFICIENCY")
    print(f"  Total energy spent (all actions): {eff_m['total_energy_spent']:,.1f}")
    print(f"  Total calories eaten:             {eff_m['total_calories_eaten']:,.1f}")
    print(f"  Caloric efficiency (cal/cost):    {eff_m['caloric_efficiency']:.3f}")
    print(f"  Move exploration ratio:           {eff_m['move_exploration_ratio']:.3f} (unique tiles / moves)")

# ── NEW: POPULATION DYNAMICS ─────────────────────────────────────────

pop_m = _compute_population_dynamics(df)
if pop_m:
    print("\n👥 POPULATION DYNAMICS")
    print(f"  Total unique agents:  {pop_m['total_unique_agents']}")
    print(f"  Peak population:      {pop_m['peak_population']}")
    print(f"  Min population:       {pop_m['min_population']}")
    print(f"  Mean population:      {pop_m['mean_population']}")
    print(f"  Estimated births:     {pop_m['estimated_births']}")
    print(f"  Estimated deaths:     {pop_m['estimated_deaths']}")
    growth = pop_m['estimated_births'] - pop_m['estimated_deaths']
    print(f"  Net growth:           {growth:+d}")

# ── NEW: LIFESPAN & FITNESS ──────────────────────────────────────────

life_m = _compute_lifespan_metrics(df)
if life_m:
    print("\n🧬 LIFESPAN & FITNESS")
    print(f"  Avg lifespan:         {life_m['avg_lifespan']:.1f} ticks")
    print(f"  Median lifespan:      {life_m['median_lifespan']:.1f} ticks")
    print(f"  Max lifespan:         {life_m['max_lifespan']} ticks")
    print(f"  Std lifespan:         {life_m['std_lifespan']:.1f}")
    print(f"  Avg peak fitness:     {life_m['avg_peak_fitness']:.2f}")
    print(f"  Max peak fitness:     {life_m['max_peak_fitness']:.2f}")
    print(f"  Avg eats/agent:       {life_m['avg_eats_per_agent']:.1f}")
    print(f"  Avg pickups/agent:    {life_m['avg_pickups_per_agent']:.1f}")
    print("\n  🏅 TOP 5 AGENTS (by peak fitness):")
    top5 = life_m['top5_agents']
    print(f"    {'Agent':>7s}  {'Lifespan':>8s}  {'Fitness':>8s}  {'Eats':>5s}  {'Pickups':>7s}  {'Age':>5s}")
    for agent_id, row in top5.iterrows():
        print(f"    {agent_id:>7d}  {int(row['lifespan']):>8d}  {row['peak_fitness']:>8.1f}  {int(row['total_eats']):>5d}  {int(row['total_pickups']):>7d}  {int(row['max_age']):>5d}")

# ── NEW: SPATIAL COVERAGE ────────────────────────────────────────────

spatial_m = _compute_spatial_metrics(df)
if spatial_m:
    print("\n🗺️  SPATIAL COVERAGE")
    print(f"  Unique tiles visited (global):  {spatial_m['unique_tiles_global']}")
    print(f"  Avg tiles per agent:            {spatial_m['avg_tiles_per_agent']:.1f}")
    print(f"  Median tiles per agent:         {spatial_m['median_tiles_per_agent']:.1f}")
    print(f"  Max tiles by one agent:         {spatial_m['max_tiles_per_agent']}")
    print(f"  Avg displacement (start→end):   {spatial_m['avg_displacement']:.1f}")
    print(f"  Max displacement:               {spatial_m['max_displacement']:.1f}")
    hx, hy = spatial_m['hotspot_tile']
    print(f"  Hotspot tile:                   ({hx}, {hy}) — {spatial_m['hotspot_visits']} visits")

# ── NEW: FARMING PIPELINE ────────────────────────────────────────────

farm_m = _compute_farming_metrics(df)
if farm_m:
    print("\n🌾 FARMING PIPELINE")
    print(f"  Seeds picked up:        {farm_m['total_seed_pickups']}")
    print(f"  Seeds planted:          {farm_m['total_seed_plants']}")
    print(f"  Plant-to-pickup ratio:  {farm_m['plant_to_pickup_ratio']:.3f}")
    print(f"  Unique seed collectors: {farm_m['unique_seed_collectors']}")
    print(f"  Unique planters:        {farm_m['unique_planters']}")
    print(f"  Food picked up:         {farm_m['total_food_pickups']}")
    print(f"  Food eaten:             {farm_m['total_eats']}")
    print(f"  Food pickups per eat:   {farm_m['food_per_eat']:.2f}")

# ── NEW: BEHAVIORAL DIVERSITY ────────────────────────────────────────

div_m = _compute_behavioral_diversity(df)
if div_m:
    print("\n🎭 BEHAVIORAL DIVERSITY")
    print(f"  Avg strategy entropy:   {div_m['avg_entropy']:.3f} / {div_m['max_possible_entropy']:.3f} ({div_m['entropy_ratio']*100:.1f}% of max)")
    print(f"  Most uniform agent:     #{div_m['min_entropy_agent']} (lowest entropy — specialist)")
    print(f"  Most diverse agent:     #{div_m['max_entropy_agent']} (highest entropy — generalist)")
    print("  Action variability across agents (std):")
    for act, std in sorted(div_m['action_std_across_agents'].items()):
        print(f"    {act:15s}: {std:.4f}")

# ── NEW: ACTION SEQUENCES (n-grams) ─────────────────────────────────

seq_m = _compute_action_sequences(df, top_n=10)
if seq_m:
    print("\n🔗 ACTION SEQUENCES (most common)")
    print("  Top 2-grams:")
    for seq, cnt in seq_m['top_bigrams']:
        print(f"    {seq:35s}  {cnt:>8,d}")
    print("  Top 3-grams:")
    for seq, cnt in seq_m['top_trigrams']:
        print(f"    {seq:50s}  {cnt:>8,d}")

# ── NEW: TEMPORAL PHASES ─────────────────────────────────────────────

phase_m = _compute_temporal_phases(df, n_phases=4)
if phase_m:
    print("\n📅 TEMPORAL PHASES (simulation split into 4 quarters)")
    for phase_idx, info in sorted(phase_m.items()):
        print(f"\n  Phase {phase_idx + 1} (ticks {info['ticks']}): {info['n_agents']} agents, {info['n_actions']:,} actions, avg energy {info['avg_energy']}")
        dist = info['action_dist']
        parts = [f"{a}={p:.1f}%" for a, p in sorted(dist.items(), key=lambda x: -x[1])]
        print(f"    {', '.join(parts)}")

# ── OVERALL VERDICT ──────────────────────────────────────────────────

print("\n🏆 OVERALL VERDICT")
within_wait_target = 32 <= wait_pct <= 35
good_eat_rate = eat_success_rate >= 3 if len(eat_actions) > 0 else False
good_survival = total_ticks >= 1500

verdict_items = []
if good_survival:
    verdict_items.append("✅ Survival: EXCELLENT" if total_ticks >= 5000 else "✅ Survival: GOOD")
else:
    verdict_items.append("❌ Survival: TOO SHORT")
if good_eat_rate:
    verdict_items.append("✅ Feeding: 100% EAT success")
else:
    verdict_items.append("⚠️  Feeding: needs improvement")
if within_wait_target:
    verdict_items.append("✅ Activity: balanced")
elif wait_pct < 32:
    verdict_items.append("⚠️  Activity: too aggressive")
else:
    verdict_items.append("⚠️  Activity: too passive")

if pop_m and pop_m['estimated_births'] > 0:
    verdict_items.append(f"✅ Reproduction: {pop_m['estimated_births']} births")
if farm_m and farm_m['total_seed_plants'] > 0:
    verdict_items.append(f"✅ Farming: {farm_m['total_seed_plants']} seeds planted by {farm_m['unique_planters']} agents")
if div_m:
    if div_m['entropy_ratio'] >= 0.5:
        verdict_items.append(f"✅ Diversity: {div_m['entropy_ratio']*100:.0f}% entropy utilization")
    else:
        verdict_items.append(f"⚠️  Diversity: low ({div_m['entropy_ratio']*100:.0f}% entropy)")

for item in verdict_items:
    print(f"  {item}")

if is_new_format:
    print("\n" + "=" * 70)
    print("🧠 WORLD MODEL TRAINING DATA ANALYSIS")
    print("=" * 70)

    if 'energy' in df.columns and 'energy_next' in df.columns:
        print("\n⚡ ENERGY DYNAMICS")
        df['energy_delta'] = df['energy_next'] - df['energy']
        print(f"  Avg energy change per action: {df['energy_delta'].mean():.3f}")
        print(f"  Total energy gained: {df[df['energy_delta'] > 0]['energy_delta'].sum():.1f}")
        print(f"  Total energy lost: {df[df['energy_delta'] < 0]['energy_delta'].sum():.1f}")

    obs_cols = [c for c in df.columns if c.startswith('obs_') and not c.startswith('obs_next')]
    obs_next_cols = [c for c in df.columns if c.startswith('obs_next_')]
    if obs_cols:
        print("\n👁️ OBSERVATION ANALYSIS")
        print(f"  Observation dimensions: {len(obs_cols)}")
        obs_variance = df[obs_cols].values.var(axis=0).mean()
        print(f"  Avg observation variance: {obs_variance:.4f}")

    print("\n📊 DATA QUALITY")
    print(f"  Total transitions: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    missing_count = df.drop(columns=['death_reason'], errors='ignore').isnull().sum().sum()
    print(f"  Missing values: {missing_count}")

    if obs_cols and obs_next_cols:
        print("\n🎓 WORLD MODEL TRAINING READINESS")
        print(f"  State dimensions: {len(obs_cols)}")
        print(f"  Next state dimensions: {len(obs_next_cols)}")
        print(f"  Action classes: {df['action'].nunique()}")
        print(f"  Transitions available: {len(df):,}")
        min_samples = 10000
        if len(df) >= min_samples:
            print(f"  ✅ Sufficient data for initial training (>{min_samples})")
        else:
            print(f"  ⚠️  Need more data ({len(df)}/{min_samples} samples)")

print("=" * 70)
