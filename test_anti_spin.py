"""
Quick test to verify anti-spinning penalties work.
"""
import sys
from agents.learning import RewardShaper
from agents.actions import Action, ActionResult
from agents.agent import Agent
from agents.genome import Genome
from world.world import World
import yaml

# Load config
with open('config/training_easy.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create minimal world (simplified - don't need full initialization)
class SimpleWorld:
    def __init__(self):
        self.width = 50
        self.height = 50
        self.objects = {}

world = SimpleWorld()

# Create test agent
genome = Genome(
    weight_count=100,
    trait_ranges={'metabolism_rate': (0.8, 1.2), 'vision_radius': (3, 7)}
)
agent = Agent(x=25, y=25, genome=genome, max_energy=1000.0)

# Create reward shaper
shaper = RewardShaper()

print("="*70)
print("TESTING ANTI-SPINNING MECHANISM")
print("="*70)

# Simulate spinning behavior
print("\n📍 Test 1: Agent spinning in place (4 turns, no movement)")
rewards = []
for i in range(4):
    action = Action.TURN_LEFT if i % 2 == 0 else Action.TURN_RIGHT
    result = ActionResult(True, 0.5, f"Turned {action.name}")
    
    reward = shaper.calculate_reward(
        action, result, agent.energy, agent.energy - 0.5, agent, world
    )
    rewards.append(reward)
    print(f"  Turn {i+1}: {action.name} -> reward = {reward:.3f}")

avg_spin_reward = sum(rewards) / len(rewards)
print(f"  Average reward while spinning: {avg_spin_reward:.3f}")
print(f"  {'❌ PENALTY APPLIED' if avg_spin_reward < 0 else '⚠️  No penalty detected'}")

# Reset shaper
shaper.reset()

print("\n📍 Test 2: Agent moving forward (exploring)")
rewards = []
for i in range(4):
    action = Action.MOVE_FORWARD
    result = ActionResult(True, 2.0, "Moved forward")
    agent.x += 1  # Simulate movement
    
    reward = shaper.calculate_reward(
        action, result, agent.energy, agent.energy - 2.0, agent, world
    )
    rewards.append(reward)
    print(f"  Move {i+1}: {action.name} -> reward = {reward:.3f}")

avg_move_reward = sum(rewards) / len(rewards)
print(f"  Average reward while moving: {avg_move_reward:.3f}")
print(f"  {'✅ EXPLORATION BONUS' if avg_move_reward > 0 else '⚠️  No bonus detected'}")

# Compare
print("\n" + "="*70)
print("📊 COMPARISON")
print("="*70)
print(f"  Spinning reward:  {avg_spin_reward:+.3f}")
print(f"  Movement reward:  {avg_move_reward:+.3f}")
print(f"  Difference:       {avg_move_reward - avg_spin_reward:+.3f}")

if avg_move_reward > avg_spin_reward:
    print("\n✅ SUCCESS: Moving is more rewarding than spinning!")
else:
    print("\n❌ FAILURE: Spinning is still better than moving!")

print("="*70)
