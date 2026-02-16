"""Analyze latest action/transition logs with compatibility for old and new schemas."""

import glob
import os
import pandas as pd


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

print("\n🏆 OVERALL VERDICT")
within_wait_target = 32 <= wait_pct <= 35
good_eat_rate = eat_success_rate >= 3 if len(eat_actions) > 0 else False
good_survival = total_ticks >= 1500

if within_wait_target and good_eat_rate and good_survival:
    print("  ✅ SUCCESS! V3.1 parameters are well-tuned!")
elif within_wait_target and good_eat_rate:
    print("  ⚠️  PARTIAL SUCCESS - Good behavior but short survival")
elif wait_pct < 30:
    print("  ❌ TOO AGGRESSIVE - Reduce exploration bonus to +0.10")
elif wait_pct > 38:
    print("  ❌ TOO PASSIVE - Increase exploration bonus to +0.20")
else:
    print("  ⚠️  MIXED RESULTS - Analyze detailed logs")

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
