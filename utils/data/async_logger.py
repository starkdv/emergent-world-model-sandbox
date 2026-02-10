"""
High-performance asynchronous logger with batched I/O.

Removes file I/O bottlenecks from the main simulation loop by using:
- Background thread for all disk operations
- Queue-based architecture for non-blocking writes
- Batch writing to minimize file open/close overhead
- Configurable buffer sizes and flush intervals

Author: Karan Vasa
Date: February 10, 2026
"""

import csv
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from agents.agent import Agent
    from agents.actions import Action, ActionResult
    from world.world import World


class AsyncWorldModelLogger:
    """
    High-performance asynchronous logger for world model training data.
    
    Uses a background thread to handle all file I/O, preventing disk operations
    from blocking the main simulation loop. Batches writes for maximum throughput.
    
    Key optimizations:
    - Non-blocking queue-based writes
    - Batch writing (configurable batch size)
    - Background thread for all I/O
    - Periodic auto-flush
    - Graceful shutdown with data integrity
    """
    
    def __init__(
        self,
        output_dir: str = "data/logs",
        log_every_n_ticks: int = 1,
        batch_size: int = 100,
        flush_interval: float = 2.0,
        queue_maxsize: int = 10000
    ):
        """
        Initialize the async logger.
        
        Args:
            output_dir: Directory to store CSV files
            log_every_n_ticks: Log world states every N ticks
            batch_size: Number of log entries to batch before writing
            flush_interval: Seconds between forced flushes
            queue_maxsize: Maximum queue size (blocks if full)
        """
        self.output_dir = output_dir
        self.log_every_n_ticks = log_every_n_ticks
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.current_tick = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.transitions_file = os.path.join(output_dir, f"transitions_{timestamp}.csv")
        self.episodes_file = os.path.join(output_dir, f"episodes_{timestamp}.csv")
        self.world_states_file = os.path.join(output_dir, f"world_states_{timestamp}.csv")
        
        # Initialize CSV files with headers
        self._init_files()
        
        # Queue for log entries: (log_type, data)
        self.log_queue = queue.Queue(maxsize=queue_maxsize)
        
        # Buffers for batch writing
        self.transitions_buffer = deque()
        self.episodes_buffer = deque()
        self.world_states_buffer = deque()
        
        # Track episode data
        self.episode_data = {}  # agent_id -> episode stats
        
        # Background thread control
        self._running = True
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        
        # Statistics
        self.stats = {
            'transitions_logged': 0,
            'episodes_logged': 0,
            'world_states_logged': 0,
            'batches_written': 0,
            'queue_overflows': 0
        }
        
        print(f"AsyncWorldModelLogger initialized (batch_size={batch_size}):")
        print(f"  Transitions: {self.transitions_file}")
        print(f"  Episodes: {self.episodes_file}")
        print(f"  World states: {self.world_states_file}")
    
    def _init_files(self) -> None:
        """Initialize CSV files with headers."""
        # Transitions header
        header = [
            'tick', 'agent_id', 'episode_step',
            'action', 'action_value', 'success', 'energy_cost',
            'x', 'y', 'direction_x', 'direction_y',
            'x_next', 'y_next', 'direction_x_next', 'direction_y_next',
            'energy', 'energy_next', 'energy_pct', 'energy_pct_next',
            'age', 'inventory_count', 'inventory_count_next',
            'fitness', 'fitness_next',
            'reward', 'done', 'death_reason',
            'tile_terrain', 'tile_fertility', 'tile_moisture',
            'tile_has_food', 'tile_has_plant', 'tile_has_seed',
            'tile_food_calories',
            'total_food_count', 'total_plant_count', 'alive_agents',
            'metabolism_rate', 'vision_radius',
        ]
        for i in range(64):
            header.append(f'obs_{i}')
        for i in range(64):
            header.append(f'obs_next_{i}')
        
        with open(self.transitions_file, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(header)
        
        # Episodes header
        with open(self.episodes_file, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'agent_id', 'generation', 'lineage_id',
                'start_tick', 'end_tick', 'duration',
                'total_reward', 'total_actions',
                'successful_eats', 'successful_pickups',
                'tiles_explored', 'final_fitness',
                'death_reason', 'final_energy',
                'max_energy_reached', 'avg_energy',
                'metabolism_rate', 'vision_radius'
            ])
        
        # World states header
        with open(self.world_states_file, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'tick',
                'alive_agents', 'total_agents',
                'total_food', 'total_plants', 'total_seeds',
                'avg_agent_energy', 'min_agent_energy', 'max_agent_energy',
                'avg_agent_age', 'max_agent_age',
                'avg_fertility', 'avg_moisture',
                'total_fitness', 'avg_fitness'
            ])
    
    def _writer_loop(self) -> None:
        """Background thread that handles all file I/O."""
        last_flush = time.time()
        
        while self._running or not self.log_queue.empty():
            try:
                # Get item from queue with timeout
                log_type, data = self.log_queue.get(timeout=0.1)
                
                # Add to appropriate buffer
                if log_type == 'transition':
                    self.transitions_buffer.append(data)
                elif log_type == 'episode':
                    self.episodes_buffer.append(data)
                elif log_type == 'world_state':
                    self.world_states_buffer.append(data)
                
                self.log_queue.task_done()
                
            except queue.Empty:
                pass
            
            # Check if we should flush
            current_time = time.time()
            should_flush = (
                len(self.transitions_buffer) >= self.batch_size or
                len(self.episodes_buffer) >= self.batch_size or
                len(self.world_states_buffer) >= self.batch_size or
                (current_time - last_flush) >= self.flush_interval
            )
            
            if should_flush:
                self._flush_buffers()
                last_flush = current_time
                self.stats['batches_written'] += 1
        
        # Final flush on exit
        self._flush_buffers()
    
    def _flush_buffers(self) -> None:
        """Write all buffered entries to disk."""
        # Flush transitions
        if self.transitions_buffer:
            with open(self.transitions_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                while self.transitions_buffer:
                    writer.writerow(self.transitions_buffer.popleft())
        
        # Flush episodes
        if self.episodes_buffer:
            with open(self.episodes_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                while self.episodes_buffer:
                    writer.writerow(self.episodes_buffer.popleft())
        
        # Flush world states
        if self.world_states_buffer:
            with open(self.world_states_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                while self.world_states_buffer:
                    writer.writerow(self.world_states_buffer.popleft())
    
    def log_transition(
        self,
        tick: int,
        agent: 'Agent',
        action: 'Action',
        result: 'ActionResult',
        reward: float,
        obs_before: np.ndarray,
        obs_after: np.ndarray,
        world: 'World',
        x_before: int,
        y_before: int,
        energy_before: float,
        done: bool = False,
        death_reason: str = ""
    ) -> None:
        """
        Queue a transition for logging (non-blocking).
        
        Args:
            tick: Current simulation tick
            agent: Agent that performed the action
            action: Action taken
            result: Result of the action
            reward: Reward signal
            obs_before: Observation vector before action
            obs_after: Observation vector after action
            world: World instance for context
            x_before, y_before: Position before action
            energy_before: Energy before action
            done: Whether episode terminated
            death_reason: Reason for death if done
        """
        from world.objects import EdibleComponent, PlantComponent, SeedComponent
        
        # Initialize episode tracking if needed
        if agent.id not in self.episode_data:
            self.episode_data[agent.id] = {
                'start_tick': tick,
                'total_reward': 0,
                'total_actions': 0,
                'successful_eats': 0,
                'successful_pickups': 0,
                'tiles_visited': set(),
                'max_energy': energy_before,
                'energy_sum': 0,
                'energy_count': 0
            }
        
        ep = self.episode_data[agent.id]
        ep['total_reward'] += reward
        ep['total_actions'] += 1
        ep['tiles_visited'].add((agent.x, agent.y))
        ep['max_energy'] = max(ep['max_energy'], agent.energy)
        ep['energy_sum'] += agent.energy
        ep['energy_count'] += 1
        
        if result.success:
            if action.name == 'EAT':
                ep['successful_eats'] += 1
            elif action.name == 'PICK_UP':
                ep['successful_pickups'] += 1
        
        # Get tile information
        tile = world.get_tile(agent.x, agent.y)
        tile_terrain = tile.terrain_type.value if tile else 0
        tile_fertility = tile.fertility if tile else 0
        tile_moisture = tile.moisture if tile else 0
        
        tile_has_food = 0
        tile_has_plant = 0
        tile_has_seed = 0
        tile_food_calories = 0
        
        if tile and tile.object_ids:
            for obj_id in tile.object_ids:
                obj = world.objects.get(obj_id)
                if obj:
                    edible = obj.get_component(EdibleComponent)
                    if edible:
                        tile_has_food = 1
                        tile_food_calories = edible.calories * edible.freshness
                    if obj.get_component(PlantComponent):
                        tile_has_plant = 1
                    if obj.get_component(SeedComponent):
                        tile_has_seed = 1
        
        # Count world objects
        total_food = sum(1 for o in world.objects.values() if o.has_component(EdibleComponent))
        total_plants = sum(1 for o in world.objects.values() if o.has_component(PlantComponent))
        alive_agents = sum(1 for a in world.agents.values() if a.alive)
        
        # Build row
        row = [
            tick, agent.id, ep['total_actions'],
            action.name, action.value, int(result.success), round(result.energy_cost, 3),
            x_before, y_before, agent.direction[0], agent.direction[1],
            agent.x, agent.y, agent.direction[0], agent.direction[1],
            round(energy_before, 2), round(agent.energy, 2),
            round(energy_before / agent.max_energy, 3), round(agent.energy / agent.max_energy, 3),
            agent.age, len(agent.inventory) - (1 if action.name == 'PICK_UP' and result.success else 0),
            len(agent.inventory),
            round(agent.fitness - 0.1, 2), round(agent.fitness, 2),
            round(reward, 4), int(done), death_reason,
            tile_terrain, round(tile_fertility, 3), round(tile_moisture, 3),
            tile_has_food, tile_has_plant, tile_has_seed, round(tile_food_calories, 2),
            total_food, total_plants, alive_agents,
            round(agent.metabolism_rate, 4), round(agent.genome.traits.get('vision_radius', 5), 2)
        ]
        
        # Add observation vectors
        row.extend([round(float(x), 5) for x in obs_before])
        row.extend([round(float(x), 5) for x in obs_after])
        
        # Queue the row (non-blocking)
        try:
            self.log_queue.put_nowait(('transition', row))
            self.stats['transitions_logged'] += 1
        except queue.Full:
            self.stats['queue_overflows'] += 1
            # Drop the log entry if queue is full
        
        # Log episode end if done
        if done:
            self._queue_episode_end(agent, tick, death_reason)
    
    def _queue_episode_end(self, agent: 'Agent', end_tick: int, death_reason: str) -> None:
        """Queue episode summary for logging."""
        if agent.id not in self.episode_data:
            return
        
        ep = self.episode_data[agent.id]
        duration = end_tick - ep['start_tick']
        avg_energy = ep['energy_sum'] / ep['energy_count'] if ep['energy_count'] > 0 else 0
        
        row = [
            agent.id, agent.genome.generation, agent.genome.lineage_id,
            ep['start_tick'], end_tick, duration,
            round(ep['total_reward'], 2), ep['total_actions'],
            ep['successful_eats'], ep['successful_pickups'],
            len(ep['tiles_visited']), round(agent.fitness, 2),
            death_reason, round(agent.energy, 2),
            round(ep['max_energy'], 2), round(avg_energy, 2),
            round(agent.metabolism_rate, 4),
            round(agent.genome.traits.get('vision_radius', 5), 2)
        ]
        
        try:
            self.log_queue.put_nowait(('episode', row))
            self.stats['episodes_logged'] += 1
        except queue.Full:
            self.stats['queue_overflows'] += 1
        
        # Clean up episode data
        del self.episode_data[agent.id]
    
    def log_world_state(self, tick: int, world: 'World') -> None:
        """Queue world state snapshot for logging."""
        if tick % self.log_every_n_ticks != 0:
            return
        
        from world.objects import EdibleComponent, PlantComponent, SeedComponent
        
        alive_agents = [a for a in world.agents.values() if a.alive]
        total_agents = len(world.agents)
        
        total_food = sum(1 for o in world.objects.values() if o.has_component(EdibleComponent))
        total_plants = sum(1 for o in world.objects.values() if o.has_component(PlantComponent))
        total_seeds = sum(1 for o in world.objects.values() if o.has_component(SeedComponent))
        
        if alive_agents:
            avg_energy = sum(a.energy for a in alive_agents) / len(alive_agents)
            min_energy = min(a.energy for a in alive_agents)
            max_energy = max(a.energy for a in alive_agents)
            avg_age = sum(a.age for a in alive_agents) / len(alive_agents)
            max_age = max(a.age for a in alive_agents)
            total_fitness = sum(a.fitness for a in alive_agents)
            avg_fitness = total_fitness / len(alive_agents)
        else:
            avg_energy = min_energy = max_energy = 0
            avg_age = max_age = 0
            total_fitness = avg_fitness = 0
        
        # Calculate average soil stats
        total_fertility = 0
        total_moisture = 0
        tile_count = 0
        for row in world.tiles:
            for tile in row:
                if tile.terrain_type.value == "soil":
                    total_fertility += tile.fertility
                    total_moisture += tile.moisture
                    tile_count += 1
        
        avg_fertility = total_fertility / tile_count if tile_count > 0 else 0
        avg_moisture = total_moisture / tile_count if tile_count > 0 else 0
        
        row = [
            tick,
            len(alive_agents), total_agents,
            total_food, total_plants, total_seeds,
            round(avg_energy, 2), round(min_energy, 2), round(max_energy, 2),
            round(avg_age, 2), max_age,
            round(avg_fertility, 4), round(avg_moisture, 4),
            round(total_fitness, 2), round(avg_fitness, 2)
        ]
        
        try:
            self.log_queue.put_nowait(('world_state', row))
            self.stats['world_states_logged'] += 1
        except queue.Full:
            self.stats['queue_overflows'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics."""
        return {
            **self.stats,
            'queue_size': self.log_queue.qsize(),
            'transitions_buffered': len(self.transitions_buffer),
            'episodes_buffered': len(self.episodes_buffer),
            'world_states_buffered': len(self.world_states_buffer),
        }
    
    def close(self) -> None:
        """Shut down logger and flush all pending writes."""
        print("\nShutting down AsyncWorldModelLogger...")
        
        # Signal shutdown
        self._running = False
        
        # Wait for queue to empty
        self.log_queue.join()
        
        # Wait for writer thread
        self._writer_thread.join(timeout=5.0)
        
        # Final flush
        self._flush_buffers()
        
        stats = self.get_stats()
        print(f"\nAsyncWorldModelLogger closed:")
        print(f"  Transitions logged: {stats['transitions_logged']}")
        print(f"  Episodes logged: {stats['episodes_logged']}")
        print(f"  World states logged: {stats['world_states_logged']}")
        print(f"  Batches written: {stats['batches_written']}")
        if stats['queue_overflows'] > 0:
            print(f"  ⚠️  Queue overflows: {stats['queue_overflows']}")
        print(f"  Files:")
        print(f"    - {self.transitions_file}")
        print(f"    - {self.episodes_file}")
        print(f"    - {self.world_states_file}")
