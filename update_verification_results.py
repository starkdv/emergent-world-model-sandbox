"""
Update V2 Final Verification Results

Run this after test_v2_final_verification.py completes.
Automatically analyzes results and updates the results document.
"""

import pandas as pd
import glob
import os
from datetime import datetime

print("=" * 70)
print("V2 FINAL VERIFICATION - RESULTS UPDATER")
print("=" * 70)

# Find the verification test log file
log_dir = 'data/logs'
action_files = glob.glob(os.path.join(log_dir, 'agent_actions_20251116_011723.csv'))

if not action_files:
    print("\n❌ Verification log file not found!")
    print("Expected: agent_actions_20251116_011723.csv")
    print("\nMake sure test_v2_final_verification.py has completed.")
    exit(1)

log_file = action_files[0]
print(f"\n📊 Analyzing: {log_file}")

# Load data
df = pd.read_csv(log_file)

# Calculate metrics
total_actions = len(df)
total_ticks = df['tick'].max()
num_agents = df['agent_id'].nunique()

print(f"\n📈 RESULTS:")
print(f"  Total ticks: {total_ticks}")
print(f"  Total actions: {total_actions:,}")
print(f"  Agents: {num_agents}")

# Action distribution
action_counts = df['action'].value_counts()
wait_pct = (action_counts.get('WAIT', 0) / total_actions * 100) if total_actions > 0 else 0

# EAT analysis
eat_actions = df[df['action'] == 'EAT']
eat_attempts = len(eat_actions)
eat_success = eat_actions['success'].sum()
eat_success_rate = (eat_success / eat_attempts * 100) if eat_attempts > 0 else 0

# Movement analysis
move_actions = df[df['action'] == 'MOVE_FORWARD']
move_attempts = len(move_actions)
move_success = move_actions['success'].sum()
move_success_rate = (move_success / move_attempts * 100) if move_attempts > 0 else 0

print(f"\n🎯 KEY METRICS:")
print(f"  Survival time: {total_ticks} ticks")
print(f"  WAIT actions: {wait_pct:.1f}%")
print(f"  EAT success: {eat_success_rate:.1f}% ({eat_success}/{eat_attempts})")
print(f"  Movement success: {move_success_rate:.1f}% ({move_success}/{move_attempts})")

# Assessment
print(f"\n📊 ASSESSMENT vs TARGETS:")

survival_target = 2000
survival_ok = total_ticks >= survival_target * 0.8
print(f"  Survival: {total_ticks} / ~{survival_target} ticks", 
      "✅" if survival_ok else "❌")

eat_target = 6.0
eat_ok = eat_success_rate >= 4.0
print(f"  EAT success: {eat_success_rate:.1f}% / ~{eat_target}%",
      "✅" if eat_ok else "❌")

wait_ok = 35 <= wait_pct <= 45
print(f"  WAIT actions: {wait_pct:.1f}% / 35-45%",
      "✅" if wait_ok else "⚠️ ")

move_ok = move_success_rate >= 70
print(f"  Movement: {move_success_rate:.1f}% / ≥70%",
      "✅" if move_ok else "❌")

# Overall verdict
all_good = survival_ok and eat_ok and move_ok
print(f"\n🏆 OVERALL VERDICT:")
if all_good:
    print("  ✅ VERIFICATION SUCCESSFUL!")
    print("  V2 Baseline performs as expected.")
    print("  System is PRODUCTION READY! 🎉")
else:
    print("  ⚠️  MIXED RESULTS")
    if not survival_ok:
        print("  - Survival time below target")
    if not eat_ok:
        print("  - EAT success below target")
    if not move_ok:
        print("  - Movement success below target")
    print("\n  Review logs for details.")

# Update results document
results_file = "V2_FINAL_VERIFICATION_RESULTS.md"
print(f"\n📝 Updating {results_file}...")

try:
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Update metrics
    content = content.replace("_pending_", f"{total_ticks}", 1)  # Survival time
    content = content.replace("_pending_", f"{eat_success_rate:.1f}%", 1)  # EAT success
    content = content.replace("_pending_", f"{wait_pct:.1f}%", 1)  # WAIT %
    content = content.replace("_pending_", f"{move_success_rate:.1f}%", 1)  # Movement
    
    # Update status markers
    if survival_ok:
        content = content.replace("⏳", "✅", 1)
    if eat_ok:
        content = content.replace("⏳", "✅", 1)
    if wait_ok:
        content = content.replace("⏳", "✅", 1)
    if move_ok:
        content = content.replace("⏳", "✅", 1)
    
    # Add timestamp
    now = datetime.now().strftime("%H:%M:%S")
    content = content.replace("_pending_", now, 1)  # Completion time
    
    with open(results_file, 'w') as f:
        f.write(content)
    
    print(f"  ✅ Results document updated!")
    
except Exception as e:
    print(f"  ⚠️  Could not update document: {e}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nDocuments updated:")
print(f"  - {results_file}")
print(f"\nNext steps:")
if all_good:
    print("  1. Review V2_FINAL_VERIFICATION_RESULTS.md")
    print("  2. Mark system as PRODUCTION READY")
    print("  3. Begin evolution + learning integration")
else:
    print("  1. Review detailed logs")
    print("  2. Investigate discrepancies")
    print("  3. Consider re-running test")
print("=" * 70)
