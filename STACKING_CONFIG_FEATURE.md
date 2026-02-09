# Object Stacking Configuration Feature

**Date**: November 16, 2025  
**Author**: Karan Vasa

## Overview

Added a configurable `allow_stacking` option that controls whether multiple objects can occupy the same tile in the simulation. This provides flexibility to switch between strict one-object-per-tile enforcement (more realistic) and allow overlapping objects (legacy behavior).

---

## Configuration

### Config Files Updated

**`config/default.yaml`**
```yaml
world:
  width: 100
  height: 100
  initial_resources: 50
  resource_spawn_rate: 0.01
  seed: null
  allow_stacking: false  # NEW: If false, enforces one object per tile
```

**`config/training_easy.yaml`**
```yaml
world:
  width: 100
  height: 50
  initial_resources: 50
  resource_spawn_rate: 0.01
  seed: null
  allow_stacking: false  # NEW: If false, enforces one object per tile
```

### Default Behavior

- **`allow_stacking: false`** (default) - **STRICT MODE**: Enforces one object per tile. If a tile is occupied, the system attempts to place objects in nearby empty tiles. If no space is available, the object is either not added or removed.

- **`allow_stacking: true`** - **LEGACY MODE**: Allows multiple objects on the same tile (plants, berries, seeds can overlap). This was the old behavior before the fix.

---

## Implementation Details

### 1. World Class (`world/world.py`)

#### Constructor Parameter
```python
def __init__(
    self,
    # ... other parameters ...
    allow_stacking: bool = False  # NEW: Controls object stacking
):
    # Store stacking configuration
    self.allow_stacking = allow_stacking
```

#### `add_object()` Method
```python
def add_object(self, obj: WorldObject) -> bool:
    """Add object with stacking check."""
    if not self.is_valid_position(obj.x, obj.y):
        return False
    
    # Check stacking configuration
    if not self.allow_stacking:
        # Enforce one-per-tile: Check if tile already has object
        tile = self.get_tile(obj.x, obj.y)
        if tile and tile.object_ids:
            # Try nearby empty tiles
            # ... placement logic ...
            return False  # If no space found
    
    # Stacking allowed OR tile is empty - add normally
    # ... add object logic ...
    return True
```

**Behavior**:
- If `allow_stacking=False` and tile occupied: Tries 8 adjacent tiles, shuffled randomly
- If no empty tile found: Returns `False` (object not added)
- If `allow_stacking=True`: Always adds object to the tile (legacy behavior)

---

### 2. Agent Actions (`agents/agent.py`)

#### DROP Action (`_drop()`)
```python
def _drop(self, world: 'World') -> ActionResult:
    """Drop object with stacking check."""
    tile = world.tiles[self.y][self.x]
    
    if world.allow_stacking or not tile.object_ids:
        # Stacking allowed OR tile empty - drop here
        tile.object_ids.append(obj_id)
        return ActionResult(True, 1.0, f"Dropped object {obj_id}")
    
    # Stacking disabled and tile occupied - try nearby
    # ... nearby placement logic ...
    
    # No space - put back in inventory
    self.inventory.append(obj_id)
    return ActionResult(False, 0.5, "No space to drop")
```

**Behavior**:
- If `allow_stacking=False` and tile occupied: Tries nearby tiles
- If no space: Returns object to inventory (doesn't drop)
- If `allow_stacking=True`: Always drops on current tile

---

#### USE Action (`_use()` - Planting Seeds)
```python
def _use(self, world: 'World') -> ActionResult:
    """Plant seed with stacking check."""
    if tile.can_support_plant():
        if world.allow_stacking or not tile.object_ids:
            # Stacking allowed OR tile empty - plant here
            tile.object_ids.append(obj_id)
            return ActionResult(True, 2.0, "Planted seed")
        else:
            # Try nearby plantable tiles
            # ... nearby planting logic ...
            return ActionResult(False, 0.5, "Cannot plant - tile occupied")
```

**Behavior**:
- If `allow_stacking=False` and tile occupied: Tries nearby plantable tiles
- If no space: Keeps seed in inventory (doesn't plant)
- If `allow_stacking=True`: Always plants on current tile

---

#### DIE Method (`die()` - Death Drops)
```python
def die(self, world: 'World') -> None:
    """Drop inventory items on death with stacking check."""
    for obj_id in self.inventory:
        tile = world.tiles[self.y][self.x]
        
        if world.allow_stacking or not tile.object_ids:
            # Drop here
            tile.object_ids.append(obj_id)
        else:
            # Try nearby tiles
            # ... nearby placement logic ...
            
            # If no space - remove from world
            if not placed:
                world.remove_object(obj_id)
```

**Behavior**:
- If `allow_stacking=False` and tile occupied: Tries nearby tiles
- If no space: **Removes object from world** (lost on death)
- If `allow_stacking=True`: All items drop on current tile

---

### 3. Main Script (`main.py`)

```python
world = World(
    # ... other parameters ...
    allow_stacking=world_cfg.get('allow_stacking', False)  # Get from config
)
```

Uses `.get('allow_stacking', False)` to default to strict mode if config missing.

---

## Testing & Verification

### Test Scenarios

1. **Initialization** (strict mode)
   - Verify no overlapping objects during world spawn
   - Check nearby placement when tiles occupied

2. **Runtime Object Addition** (strict mode)
   - Seeds dropped from decomposed berries
   - Berries spawned by mature plants
   - Safety spawner adding resources

3. **Agent Actions** (strict mode)
   - DROP: Should place nearby or return to inventory
   - USE (planting): Should plant nearby or keep seed
   - Death drops: Should scatter items or remove if no space

4. **Legacy Mode** (allow_stacking=true)
   - Multiple objects should stack on same tile
   - No nearby placement attempts

### Visual Inspection

Run GUI and verify:
```bash
python main.py --config config/training_easy.yaml --gui
```

**Look for**:
- No overlapping objects (strict mode)
- Objects properly spaced
- Console messages about nearby placement
- Agent behavior adapting to space constraints

---

## Benefits

### Strict Mode (`allow_stacking: false`)

✅ **More Realistic**
- Mimics physical space constraints
- Objects can't overlap in real world

✅ **Better Visualization**
- Clear separation of objects
- Easier to see what's on each tile

✅ **Strategic Gameplay**
- Agents must consider space when dropping/planting
- Encourages efficient inventory management

✅ **Prevents Bugs**
- No invisible stacked objects
- Clearer world state

### Legacy Mode (`allow_stacking: true`)

✅ **Backward Compatibility**
- Preserves old simulation behavior
- Useful for comparing results

✅ **Higher Density**
- More objects in smaller worlds
- Useful for testing resource abundance

✅ **Simpler Logic**
- No spatial competition
- Faster object placement

---

## Migration Guide

### Enabling Strict Mode (Recommended)

**Step 1**: Update config file
```yaml
world:
  allow_stacking: false
```

**Step 2**: Test your simulation
```bash
python main.py --config config/your_config.yaml --gui
```

**Step 3**: Monitor for issues
- Check if agents struggle to drop items
- Verify plant growth isn't blocked
- Watch for "no space" messages

### Reverting to Legacy Mode

If issues arise, temporarily enable stacking:
```yaml
world:
  allow_stacking: true
```

This restores the old behavior while you investigate.

---

## Known Behaviors

### Strict Mode Considerations

1. **Death Drops Can Be Lost**
   - If agent dies on crowded tile with full inventory
   - Items may be removed if no nearby space
   - This is realistic but may affect loot scarcity

2. **Planting May Fail**
   - Seeds can't be planted in crowded areas
   - Agents will keep trying or move elsewhere
   - Encourages spatial planning

3. **Safety Spawner Impact**
   - Safety resources may fail to spawn if world is full
   - Monitor resource counts in crowded simulations
   - May need to increase `min_resources` threshold

### Legacy Mode Considerations

1. **Visual Overlap**
   - Multiple objects render on same tile
   - Can hide underlying objects
   - Makes it harder to count resources

2. **Unrealistic Density**
   - Unlimited stacking can create "magic tiles"
   - Less strategic gameplay
   - May affect learning dynamics

---

## Configuration Recommendations

### Training Scenarios
```yaml
allow_stacking: false  # Teach agents spatial awareness
```

### High-Density Testing
```yaml
allow_stacking: true   # Pack more objects in small world
```

### Production Simulations
```yaml
allow_stacking: false  # Realistic, strategic gameplay
```

### Debugging/Analysis
```yaml
allow_stacking: true   # Simplify object placement issues
```

---

## Future Enhancements

### Potential Additions

1. **Configurable Stack Limits**
   ```yaml
   max_objects_per_tile: 3  # Allow limited stacking
   ```

2. **Object Type Rules**
   ```yaml
   stacking_rules:
     plants: false      # Plants can't stack
     berries: true      # Berries can pile up (max 3)
     seeds: true        # Seeds can cluster
   ```

3. **Tile Capacity System**
   ```yaml
   tile_capacity: 5.0  # Objects have size/weight
   berry_size: 1.0
   plant_size: 3.0
   ```

4. **Agent Size/Space**
   ```yaml
   agents_block_tiles: true  # Agents take up tile space
   ```

---

## Code Locations

### Files Modified

1. **`config/default.yaml`** - Added `allow_stacking` setting
2. **`config/training_easy.yaml`** - Added `allow_stacking` setting
3. **`world/world.py`** - Added parameter and logic to `__init__()` and `add_object()`
4. **`agents/agent.py`** - Updated `_drop()`, `_use()`, and `die()` methods
5. **`main.py`** - Pass config value to World constructor

### Key Methods

- `World.__init__()` - Store configuration
- `World.add_object()` - Check before adding
- `Agent._drop()` - Check before dropping
- `Agent._use()` - Check before planting
- `Agent.die()` - Check before death drops

---

## Summary

The `allow_stacking` configuration provides a flexible system for controlling object placement behavior:

- **Default (`false`)**: Strict one-object-per-tile for realistic, strategic gameplay
- **Optional (`true`)**: Allow stacking for legacy compatibility or high-density testing

This feature improves simulation realism while maintaining backward compatibility and providing flexibility for different use cases.

---

**Status**: ✅ **Fully Implemented**  
**Testing**: ⏳ **Pending Full Simulation Run**  
**Documentation**: ✅ **Complete**
