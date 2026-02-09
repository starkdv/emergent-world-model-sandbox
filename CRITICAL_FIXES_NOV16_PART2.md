# Critical Fixes - Part 2: Agent Behavior & World State
**Date**: November 16, 2025  
**Status**: ✅ Implemented

---

## Issues Identified & Fixed

### Issue 1: ❌ Agents Picking Up ANY Object (Not Just Food)

**Problem**: Agents were picking up seeds, plants, and other non-edible objects, filling their inventory with useless items instead of food.

**Root Cause**:
```python
# OLD CODE (agents/agent.py, line ~283)
def _pick_up(self, world: 'World') -> ActionResult:
    # Pick up first object (ANY object!)
    obj_id = tile.object_ids[0]
```

**Solution**: Implemented smart pickup that prioritizes food:

```python
# NEW CODE (agents/agent.py, lines ~275-314)
def _pick_up(self, world: 'World') -> ActionResult:
    """Pick up object from current tile, prioritizing food."""
    # ... 
    
    # 1. FIRST: Look for edible items (berries)
    for obj_id in tile.object_ids:
        obj = world.objects.get(obj_id)
        if obj and obj.has_component(EdibleComponent):
            obj_id_to_pick = obj_id
            break
    
    # 2. SECOND: Look for seeds (useful for planting)
    if obj_id_to_pick is None:
        for obj_id in tile.object_ids:
            obj = world.objects.get(obj_id)
            if obj and obj.has_component(SeedComponent):
                obj_id_to_pick = obj_id
                break
    
    # 3. LAST: Pick any remaining object
    if obj_id_to_pick is None:
        obj_id_to_pick = tile.object_ids[0]
```

**Impact**: 
- ✅ Agents now fill inventory with FOOD first
- ✅ More efficient eating behavior
- ✅ Better survival rates

---

### Issue 2: ❌ Energy Declining Despite Agents Having Food

**Problem**: Agents were starving to death while carrying food in their inventory. They weren't eating frequently enough.

**Root Cause**: Agents relied purely on neural network policy to decide when to eat. The policy wasn't trained well enough to recognize "I have food + I'm hungry = EAT NOW".

**Solution**: Implemented **automatic eating** when energy drops below 40%:

```python
# NEW CODE (agents/agent.py, lines ~145-161)
# AUTO-EAT: If energy is low and agent has food, eat automatically
# This ensures agents don't starve while holding food
from world.objects import EdibleComponent
if self.energy < self.max_energy * 0.4 and self.inventory:  # Below 40% energy
    has_food = any(
        world.objects.get(obj_id) and 
        world.objects.get(obj_id).has_component(EdibleComponent)
        for obj_id in self.inventory
    )
    if has_food:
        # Force EAT action when hungry with food
        action = Action.EAT
    else:
        action = self.brain.decide(observation, epsilon=self.epsilon)
else:
    action = self.brain.decide(observation, epsilon=self.epsilon)
```

**Behavior**:
- Energy > 40%: Agent uses learned policy (might save food)
- Energy < 40% + has food: **FORCED EAT** (survival priority)
- Energy < 40% + no food: Search for food via policy

**Impact**:
- ✅ Prevents starvation while holding food
- ✅ Energy maintained at healthier levels
- ✅ Agents still learn optimal eating timing above 40% threshold

---

### Issue 3: ✅ Eating Only from Inventory (Confirmed Correct)

**Status**: No fix needed - working as designed

**How it works**:
1. Agent picks up food (berry) from tile → goes to inventory
2. Agent executes EAT action → consumes food from inventory
3. Food is removed from world entirely

**Code** (agents/agent.py, lines ~330-345):
```python
def _eat(self, world: 'World') -> ActionResult:
    """Consume edible object from inventory."""
    if not self.inventory:
        return ActionResult(False, 0.5, "Nothing to eat")
    
    # Find edible item in inventory
    for obj_id in self.inventory:
        obj = world.objects.get(obj_id)
        if obj and obj.has_component(EdibleComponent):
            # Consume the food
            energy_gained = edible.calories * edible.freshness
            self.energy = min(self.max_energy, self.energy + energy_gained)
            
            # Remove from inventory AND world
            self.inventory.remove(obj_id)
            world.remove_object(obj_id)
            
            return ActionResult(True, 1.0, f"Ate food, gained {energy_gained:.1f} energy")
```

**Why this is correct**:
- ✅ Realistic: Animals don't eat while walking, they pick up food then eat it
- ✅ Inventory management: Agents must decide what to carry
- ✅ Strategic: Can stockpile food for later (with freshness decay)

---

### Issue 4: ❌ Multiple Objects Per Tile (Trees/Food Stacking)

**Problem**: World initialization was spawning multiple plants, berries, or seeds on the same tile, creating visual clutter and unrealistic density.

**Root Cause**:
```python
# OLD CODE (main.py, lines ~228-260)
# Add plants
for _ in range(initial_resources // 2):
    x = random.randint(0, world.width - 1)
    y = random.randint(0, world.height - 1)
    # NO CHECK if tile already occupied!
    if tile and tile.is_plantable():
        plant = WorldObject(x, y)
        # ...
        world.add_object(plant)

# Same problem for berries and seeds - no collision detection!
```

**Solution**: Track occupied tiles and prevent stacking:

```python
# NEW CODE (main.py, lines ~223-300)
# Track occupied tiles to prevent multiple objects per tile
occupied_tiles = set()

# Add plants
plants_added = 0
attempts = 0
max_attempts = initial_resources * 10  # Prevent infinite loop

while plants_added < initial_resources // 2 and attempts < max_attempts:
    attempts += 1
    x = random.randint(0, world.width - 1)
    y = random.randint(0, world.height - 1)
    
    # Skip if tile already has an object
    if (x, y) in occupied_tiles:
        continue
    
    tile = world.get_tile(x, y)
    if tile and tile.is_plantable():
        plant = WorldObject(x, y)
        # ... create plant ...
        world.add_object(plant)
        occupied_tiles.add((x, y))  # Mark as occupied
        plants_added += 1

# Same logic applied to berries and seeds
```

**Impact**:
- ✅ **One object per tile maximum** during initialization
- ✅ Cleaner visual representation
- ✅ More realistic world layout
- ✅ Agents can navigate more easily

**Note**: Runtime spawning (seeds dropping, berries growing) may still create multiple objects per tile - this is acceptable as it's a natural process.

---

## Additional Enhancements

### 🔧 Anti-Spinning Penalties (From Previous Session)

**Changes** (agents/learning.py):
```python
# Track recent actions and position
self.last_actions = []  # Last 6 actions
self.last_position = None  # Last (x, y) position

# Detect spinning (3+ turns in 4 actions with no movement)
if len(self.last_actions) >= 4:
    recent = self.last_actions[-4:]
    turn_count = sum(1 for a in recent if a in [Action.TURN_LEFT, Action.TURN_RIGHT])
    move_count = sum(1 for a in recent if a == Action.MOVE_FORWARD)
    
    if turn_count >= 3 and move_count == 0:
        reward -= 0.5  # Penalty for spinning behavior

# Reward successful movement
if action == Action.MOVE_FORWARD and action_result.success:
    reward += 0.3  # Exploration bonus

# Penalty for being stuck (not moving when not intentional)
if current_position == self.last_position:
    if action not in [Action.WAIT, Action.PICK_UP, Action.EAT, Action.USE]:
        reward -= 0.2  # Stuck penalty
```

### 📊 Exploration Rate Update

**Config** (training_easy.yaml):
```yaml
learning:
  epsilon: 0.2  # 20% random actions (up from 0.12)
```

**Agent** (agents/agent.py):
```python
self.epsilon = 0.20  # Matches config
```

**Rationale**: Higher exploration with anti-spin penalties prevents agents from:
- Getting stuck in local optima
- Spinning forever
- Not discovering food-finding strategies

---

## Testing Recommendations

### Test 1: Food Pickup Priority
```bash
python main.py --learning --config config/training_easy.yaml
```

**Look for**: Agents picking up berries before seeds in debug logs

### Test 2: Auto-Eating Behavior
**Look for**: 
- Agents eating when energy drops below 40%
- Energy levels stabilizing around 40-60% range
- Fewer deaths from starvation with food in inventory

### Test 3: One Object Per Tile
**Look for**:
- Console output: "Resources added: X objects (plants: Y, berries: Z, seeds: W)"
- Visual inspection: No tile overlaps during world initialization

### Test 4: Anti-Spinning
**Look for**:
- Less circling behavior in GUI
- More forward movement
- Better exploration patterns

---

## Expected Improvements

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| **Food pickup success** | Picking up seeds/plants | 90%+ food pickups |
| **Starvation with food** | Common | Rare/Never |
| **Energy maintenance** | Declining to 0 | Stable 40-70% |
| **Spinning behavior** | Frequent | Reduced 60-80% |
| **Objects per tile** | 2-5 stacked | 1 max (init) |
| **Survival time** | ~500-1000 ticks | 1500+ ticks |

---

## Code Files Modified

1. **`agents/agent.py`**
   - Lines ~275-314: Smart pickup (prioritize food)
   - Lines ~145-161: Auto-eat when energy < 40%
   - Line ~109: Epsilon = 0.20

2. **`main.py`**
   - Lines ~223-300: One object per tile during initialization

3. **`agents/learning.py`** (Previous session)
   - Lines ~103-105: Track actions and position
   - Lines ~205-230: Anti-spinning penalties

4. **`config/training_easy.yaml`**
   - Line ~112: epsilon: 0.2

---

## Status: ✅ Ready for Testing

All 4 critical issues have been addressed:
1. ✅ Agents prioritize food pickup
2. ✅ Auto-eat prevents starvation with food
3. ✅ Eating from inventory confirmed working
4. ✅ One object per tile maximum (initialization)

Plus anti-spinning penalties from previous session.

**Next Steps**: Run simulation and monitor agent behavior!
