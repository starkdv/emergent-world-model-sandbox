"""
Main entry point for the Emergent World-Model Sandbox.

This script initializes and runs the simulation with the specified configuration.

Author: Karan Vasa
"""

import argparse
import sys
import random
from pathlib import Path

import yaml

from world.world import World
from world.objects import WorldObject, EdibleComponent, SeedComponent, PlantComponent
from world.tiles import TerrainType
from world.object_registry import ObjectRegistry, register_builtin_objects
from agents import Agent, Genome, Brain, create_default_trait_config
from utils.render import ConsoleRenderer
from utils.ui.pygame_renderer import PygameRenderer


def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
        
    Author: Karan Vasa
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """
    Main function to parse arguments and start the simulation.
    
    Returns:
        Exit code (0 for success, 1 for error)
        
    Author: Karan Vasa
    """
    parser = argparse.ArgumentParser(
        description="Emergent World-Model Sandbox - Evolution Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run with default config
  python main.py --config custom.yaml      # Run with custom config
  python main.py --no-viz                  # Run without visualization
  python main.py --gui                     # Run with Pygame GUI
  python main.py --gui --gpu               # Run with GPU isometric renderer
  python main.py --seed 42 --demo          # Run demo with specific seed        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file (default: config/default.yaml)'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization (run headless)'
    )
    
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Enable GUI visualization with Pygame (default: False)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--generations',
        type=int,
        default=None,
        help='Number of generations to run (overrides config)'    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run a quick demo of the world system'
    )
    
    parser.add_argument(
        '--log',
        action='store_true',
        help='Enable CSV logging of agent actions and states'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='data/logs',
        help='Directory for CSV log files (default: data/logs)'
    )
    
    parser.add_argument(
        '--log-frequency',
        type=int,
        default=1,
        help='Log agent states every N ticks (default: 1, every tick)'
    )
    
    parser.add_argument(
        '--world-model-log',
        action='store_true',
        help='Enable enhanced world model training data logging (transitions, episodes, world states)'
    )
    
    parser.add_argument(
        '--learning',
        action='store_true',
        help='Enable reinforcement learning for agents (helps them find food and survive)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='Learning rate for agent neural networks (default: 0.01)'
    )
    
    parser.add_argument(
        '--load-weights',
        type=str,
        default=None,
        help='Load pre-trained weights from file (e.g., data/weights/best_weights.npz)'
    )
    
    parser.add_argument(
        '--save-weights',
        action='store_true',
        help='Save best agent weights at end of run'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU-accelerated isometric renderer (requires --gui, needs moderngl)'
    )

    parser.add_argument(
        '--objects',
        type=str,
        default=None,
        help='Path to custom object definitions YAML file (e.g., config/custom_objects.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print(f"Loading configuration from {args.config}...")
        config = load_config(args.config)
          # Override config with command-line arguments
        if args.no_viz:
            config['visualization']['enabled'] = False
        
        if args.seed is not None:
            config['world']['seed'] = args.seed
        
        if args.generations is not None:
            config['simulation']['max_generations'] = args.generations
        
        if args.verbose:
            config['logging']['level'] = 'DEBUG'
        print("Configuration loaded successfully!")
        print(f"World size: {config['world']['width']}x{config['world']['height']}")
        print(f"Initial population: {config['agents']['initial_population']}")
        print(f"Max generations: {config['simulation']['max_generations']}")
        print(f"Visualization: {'Enabled' if config['visualization']['enabled'] else 'Disabled'}")
        if args.log:
            print(f"Logging: Enabled (frequency: every {args.log_frequency} tick(s))")
        if args.learning:
            print(f"Learning: ENABLED (rate: {args.learning_rate}) - Agents will learn to survive!")
        if args.load_weights:
            print(f"Pre-trained weights: Loading from {args.load_weights}")
        if args.save_weights:
            print(f"Weight saving: Enabled (will save best agents at end)")
        
        # Initialize agent logger if requested
        agent_logger = None
        if args.log:
            from utils.data import AgentLogger
            agent_logger = AgentLogger(args.log_dir, args.log_frequency)
            Agent.logger = agent_logger  # Set class-level logger (Agent already imported at top)
        
        # Initialize world model logger if requested
        world_model_logger = None
        if args.world_model_log:
            from utils.data.async_logger import AsyncWorldModelLogger
            world_model_logger = AsyncWorldModelLogger(
                output_dir=args.log_dir,
                log_every_n_ticks=args.log_frequency,
                batch_size=100,
                flush_interval=2.0
            )
            Agent.world_model_logger = world_model_logger
            print(f"World Model Logging: Enabled (async mode for high performance)")

        
        # Create world
        print("\n" + "="*60)
        print("INITIALIZING WORLD")
        print("="*60)
        
        world_cfg = config['world']
        terrain_cfg = config['terrain']
        plant_cfg = config['plants']
        resource_cfg = config['resources']
        soil_cfg = config['soil']
        learning_cfg = config.get('learning', {})
        
        # Initialize object registry BEFORE world creation so sand objects
        # can be spawned on SAND terrain tiles during terrain generation.
        register_builtin_objects()

        # Apply sand config overrides from YAML (if present)
        sand_cfg = config.get('sand', {})
        if sand_cfg:
            sand_defn = ObjectRegistry.get('sand')
            if sand_defn and sand_defn.tile_effect:
                te = sand_defn.tile_effect
                te.spread_interval = sand_cfg.get('spread_interval', te.spread_interval)
                te.spread_chance = sand_cfg.get('spread_chance', te.spread_chance)
                te.spread_radius = sand_cfg.get('spread_radius', te.spread_radius)
                te.spread_blocked_by = sand_cfg.get('spread_blocked_by', te.spread_blocked_by)
                te.germination_multiplier = sand_cfg.get('germination_multiplier', te.germination_multiplier)
                te.growth_multiplier = sand_cfg.get('growth_multiplier', te.growth_multiplier)
                te.spawn_rate_multiplier = sand_cfg.get('spawn_rate_multiplier', te.spawn_rate_multiplier)
                te.fertility_override = sand_cfg.get('fertility_override', te.fertility_override)
                te.moisture_override = sand_cfg.get('moisture_override', te.moisture_override)
                te.reclaim_terrain = sand_cfg.get('reclaim_terrain', te.reclaim_terrain)
                te.reclaim_interval = sand_cfg.get('reclaim_interval', te.reclaim_interval)
                print(f"Sand tuning: interval={te.spread_interval}, chance={te.spread_chance}, radius={te.spread_radius}, reclaim={te.reclaim_terrain}@{te.reclaim_interval}")
        
        # Load custom object definitions from config if present
        if 'objects' in config:
            loaded = ObjectRegistry.load_from_config(config['objects'])
            print(f"Loaded {loaded} custom object definitions from config")
        
        # Load custom objects from a separate YAML file (--objects flag)
        if args.objects:
            objects_path = Path(args.objects)
            if objects_path.exists():
                with open(objects_path, 'r') as f:
                    objects_data = yaml.safe_load(f)
                if objects_data and 'objects' in objects_data:
                    loaded = ObjectRegistry.load_from_config(objects_data['objects'])
                    print(f"Loaded {loaded} custom object definitions from {args.objects}")
            else:
                print(f"Warning: Objects file not found: {args.objects}")
        
        world = World(
            width=world_cfg['width'],
            height=world_cfg['height'],
            seed=world_cfg['seed'],
            soil_ratio=terrain_cfg['soil_ratio'],
            rock_ratio=terrain_cfg['rock_ratio'],
            water_ratio=terrain_cfg['water_ratio'],
            sand_ratio=terrain_cfg.get('sand_ratio', 0.05),
            fertility_range=tuple(terrain_cfg['fertility_range']),
            moisture_range=tuple(terrain_cfg['moisture_range']),
            # System configuration parameters
            plant_mature_age=plant_cfg['mature_age'],
            plant_max_age=plant_cfg['max_age'],
            decay_rate=resource_cfg['berry_freshness_decay'],
            seed_drop_chance=resource_cfg['seed_drop_chance'],
            germination_success_rate=plant_cfg['germination_success_rate'],
            fertility_consumption=plant_cfg['fertility_consumption_per_tick'],
            moisture_consumption=plant_cfg['moisture_consumption_per_tick'],
            fertility_recovery_rate=soil_cfg['fertility_recovery_rate'],
            moisture_evaporation_rate=soil_cfg['moisture_evaporation_rate'],
            moisture_recovery_rate=soil_cfg['moisture_recovery_rate'],
            fertility_return_on_death=soil_cfg['fertility_return_on_death'],
            berry_calories=resource_cfg['berry_calories'],
            safety_spawn_rate=world_cfg['resource_spawn_rate'],
            min_resources=resource_cfg.get('min_resources', 20),  # From config, default 20
            seed_max_age=plant_cfg['seed_max_age'],
            allow_stacking=world_cfg.get('allow_stacking', False),  # NEW: Get from config
            learning_train_interval_ticks=learning_cfg.get('train_interval_ticks', 3),
            learning_max_updates_per_tick=learning_cfg.get('max_updates_per_tick', 16),
            learning_enable_stagger=learning_cfg.get('stagger_updates', True),
            learning_adaptive_budget=learning_cfg.get('adaptive_budget', True),
            learning_min_updates_per_tick=learning_cfg.get('min_updates_per_tick', 2),
            learning_max_budget_updates_per_tick=learning_cfg.get('max_budget_updates_per_tick', 24),
            learning_budget_adjust_step=learning_cfg.get('budget_adjust_step', 1),
            learning_budget_high_frame_factor=learning_cfg.get('budget_high_frame_factor', 1.10),
            learning_budget_low_frame_factor=learning_cfg.get('budget_low_frame_factor', 0.80),
            parallel=config.get('simulation', {}).get('parallel', True),
        )
        
        print(f"World created: {world.width}x{world.height}")
        print(f"Seed: {world.seed}")
        print(f"Sand tiles: {sum(1 for row in world.tiles for t in row if t.terrain_type == TerrainType.SAND)}")
        
        # Add initial resources
        print("\nPopulating world with resources...")
        initial_resources = world_cfg['initial_resources']
        
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
                plant = ObjectRegistry.create(
                    "berry_plant", x, y,
                    mature_age=plant_cfg['mature_age'],
                    plant_max_age=plant_cfg['max_age'],
                    spawn_rate=plant_cfg['seed_spawn_rate'],
                )
                world.add_object(plant)
                occupied_tiles.add((x, y))
                plants_added += 1
        
        # Add berries
        berries_added = 0
        attempts = 0
        
        while berries_added < initial_resources // 4 and attempts < max_attempts:
            attempts += 1
            x = random.randint(0, world.width - 1)
            y = random.randint(0, world.height - 1)
            
            # Skip if tile already has an object
            if (x, y) in occupied_tiles:
                continue
            
            if world.is_valid_position(x, y):
                berry = ObjectRegistry.create(
                    "berry", x, y,
                    calories=config['resources']['berry_calories'],
                )
                world.add_object(berry)
                occupied_tiles.add((x, y))
                berries_added += 1
        
        # Add seeds
        seeds_added = 0
        attempts = 0
        
        while seeds_added < initial_resources // 4 and attempts < max_attempts:
            attempts += 1
            x = random.randint(0, world.width - 1)
            y = random.randint(0, world.height - 1)
            
            # Skip if tile already has an object
            if (x, y) in occupied_tiles:
                continue
            
            if world.is_valid_position(x, y):
                seed = ObjectRegistry.create(
                    "berry_seed", x, y,
                    grow_time=config['plants']['growth_time'],
                    seed_max_age=config['plants']['seed_max_age'],
                )
                world.add_object(seed)
                occupied_tiles.add((x, y))
                seeds_added += 1
        
        print(f"Resources added: {len(world.objects)} objects (plants: {plants_added}, berries: {berries_added}, seeds: {seeds_added})")

        # Spawn custom objects that have a spawn.initial_count > 0
        custom_spawned = 0
        for defn in ObjectRegistry.all_definitions().values():
            if defn.spawn.initial_count <= 0:
                continue
            # Skip builtins (already spawned above)
            if defn.type_id in ('berry', 'berry_seed', 'berry_plant', 'fertilizer', 'sand'):
                continue
            placed = 0
            attempts = 0
            target = defn.spawn.initial_count
            max_att = target * 15
            while placed < target and attempts < max_att:
                attempts += 1
                cx = random.randint(0, world.width - 1)
                cy = random.randint(0, world.height - 1)
                if (cx, cy) in occupied_tiles:
                    continue
                ctile = world.get_tile(cx, cy)
                if ctile is None:
                    continue
                # Terrain filter
                ok = False
                if defn.spawn.terrain == 'soil':
                    ok = ctile.terrain_type == TerrainType.SOIL
                elif defn.spawn.terrain == 'sand':
                    ok = ctile.terrain_type == TerrainType.SAND
                elif defn.spawn.terrain == 'plantable':
                    ok = ctile.is_plantable()
                elif defn.spawn.terrain == 'any':
                    ok = ctile.terrain_type != TerrainType.ROCK
                else:
                    ok = ctile.terrain_type == TerrainType.SOIL
                if not ok:
                    continue
                custom_obj = ObjectRegistry.create(defn.type_id, cx, cy)
                if world.add_object(custom_obj):
                    occupied_tiles.add((cx, cy))
                    placed += 1
            if placed > 0:
                print(f"  Spawned {placed}x {defn.display_name} ({defn.type_id})")
                custom_spawned += placed
        if custom_spawned > 0:
            print(f"Custom objects spawned: {custom_spawned}")
        
        # Add initial agents
        print("\nSpawning initial agent population...")
        agent_cfg = config['agents']
        initial_population = agent_cfg['initial_population']
        
        # Get brain configuration
        brain_cfg = config['brain']
        weight_count = Brain.calculate_weight_count(
            input_size=brain_cfg['input_size'],
            encoder_layers=brain_cfg['encoder_layers'],
            gru_hidden_size=brain_cfg['gru_hidden_size'],
            output_size=brain_cfg['output_size']
        )
        
        # Create trait configuration
        trait_config = create_default_trait_config()
        
        # Load pre-trained weights if requested
        pretrained_weights = None
        if args.load_weights:
            from utils.agents import BestAgentTracker
            pretrained_weights = BestAgentTracker.load_best_weights(args.load_weights)
            if pretrained_weights is not None:
                print(f"Loaded pre-trained weights ({len(pretrained_weights)} values)")
        
        # Initialize best agent tracker if saving weights
        best_agent_tracker = None
        if args.save_weights:
            from utils.agents import BestAgentTracker
            best_agent_tracker = BestAgentTracker()
        
        # Spawn agents on passable tiles
        agents_spawned = 0
        attempts = 0
        max_attempts = initial_population * 10
        
        while agents_spawned < initial_population and attempts < max_attempts:
            x = random.randint(0, world.width - 1)
            y = random.randint(0, world.height - 1)
            tile = world.get_tile(x, y)
            
            if tile and tile.is_passable():
                # Create random genome
                genome = Genome.random(weight_count, trait_config)
                  # Create agent
                agent = Agent(
                    x=x,
                    y=y,
                    genome=genome,
                    max_energy=agent_cfg['max_energy'],
                    max_age=agent_cfg['max_age'],
                    inventory_size=agent_cfg['inventory_size'],
                    metabolism_rate=agent_cfg['metabolism_rate']
                )
                
                # Initialize with pre-trained weights if available
                if pretrained_weights is not None:
                    from utils.agents import BestAgentTracker
                    BestAgentTracker.initialize_agent_from_weights(
                        agent, 
                        pretrained_weights,
                        mutation_rate=0.02  # Small mutation for diversity
                    )
                
                # Enable learning if requested
                if args.learning:
                    agent.enable_learning(
                        learning_rate=args.learning_rate,
                        discount_factor=0.95,
                        batch_size=16,
                        buffer_capacity=1000,
                        compute_backend=learning_cfg.get('compute_backend', 'auto'),
                        compute_device=learning_cfg.get('compute_device', 'auto'),
                    )
                
                world.add_agent(agent)
                agents_spawned += 1
            
            attempts += 1
        print(f"Agents spawned: {len(world.agents)} agents")

        if args.learning and world.agents:
            sample_agent = next(iter(world.agents.values()))
            if sample_agent.learner is not None:
                print(
                    f"Learning backend: {sample_agent.learner.compute_backend} "
                    f"(device: {sample_agent.learner.compute_device})"
                )
                print(
                    "Learning scheduler: "
                    f"interval={world.learning_train_interval_ticks}, "
                    f"budget={world.learning_max_updates_per_tick}/tick, "
                    f"adaptive={'on' if world.learning_adaptive_budget else 'off'}"
                )
        
        if agents_spawned < initial_population:
            print(f"Warning: Only spawned {agents_spawned}/{initial_population} agents (not enough passable tiles)")        # Set up reproduction configuration from config file
        if 'reproduction' in config:
            world.reproduction_config = config['reproduction']
            if config['reproduction'].get('enabled', False):
                print(f"\nReproduction enabled:")
                print(f"  Energy threshold: {config['reproduction'].get('energy_threshold', 0.6)*100:.0f}% of max energy")
                print(f"  Minimum age: {config['reproduction'].get('min_age', 100)} ticks")
                print(f"  Energy split: {config['reproduction'].get('energy_split', 0.6)*100:.0f}% (parent loses this much)")
                print(f"  Mutation std: {config['reproduction'].get('mutation_std', 0.02)}")
                print(f"  Cooldown: {config['reproduction'].get('cooldown_ticks', 50)} ticks")
                max_pop = config['reproduction'].get('max_population', None)
                print(f"  Max population: {max_pop if max_pop else 'unlimited'}")
        
        # Set up calamity configuration from config file
        if 'calamity' in config:
            world.calamity_config = config['calamity']
            if config['calamity'].get('enabled', False):
                print(f"\nCalamity system enabled:")
                print(f"  Interval: {config['calamity'].get('interval', 500)} ticks")
                print(f"  Destruction rate: {config['calamity'].get('destruction_rate', 0.3)*100:.0f}%")
                print(f"  Affects plants: {config['calamity'].get('affect_plants', True)}")
                print(f"  Affects food: {config['calamity'].get('affect_food', True)}")
                print(f"  Affects seeds: {config['calamity'].get('affect_seeds', False)}")
        
        # Run demo if requested
        if args.demo:
            print("\n" + "="*60)
            print("DEMO MODE - Showing world visualization")
            print("="*60 + "\n")
            
            renderer = ConsoleRenderer(world)
            renderer.print()
            print("\n" + "="*60)
            print("Demo complete!")
            print("="*60)
            print("\nRun with --gui to start the GUI visualization")
            print("(Agent system and simulation runner coming next)")
            
            # Close logger if enabled
            if agent_logger:
                agent_logger.close()
            
            return 0
        
        # Run GUI if requested
        if args.gui:
            print("\n" + "="*60)
            print("STARTING GUI VISUALIZATION")
            print("="*60 + "\n")
            
            viz_cfg = config['visualization']

            if args.gpu:
                try:
                    from utils.ui.gpu_renderer import IsometricRenderer
                    print("Using GPU-accelerated isometric renderer (ModernGL)")
                    renderer = IsometricRenderer(
                        world=world,
                        window_width=viz_cfg.get('window_width', 1200),
                        window_height=viz_cfg.get('window_height', 800),
                        tile_size=viz_cfg.get('tile_size', 20),
                        target_fps=viz_cfg.get('target_fps', 60),
                    )
                except ImportError as e:
                    print(f"GPU renderer unavailable ({e}), falling back to Pygame")
                    renderer = PygameRenderer(
                        world=world,
                        window_width=viz_cfg.get('window_width', 1200),
                        window_height=viz_cfg.get('window_height', 800),
                        tile_size=viz_cfg.get('tile_size', 20),
                        target_fps=viz_cfg.get('target_fps', 60)
                    )
            else:
                renderer = PygameRenderer(
                    world=world,
                    window_width=viz_cfg.get('window_width', 1200),
                    window_height=viz_cfg.get('window_height', 800),
                    tile_size=viz_cfg.get('tile_size', 20),
                    target_fps=viz_cfg.get('target_fps', 60)
                )
            renderer.run()
            
            # Save best agent weights if requested
            if best_agent_tracker is not None:
                print("\nSaving best agent weights...")
                # Track all surviving agents
                for agent in world.agents.values():
                    if agent.alive:
                        best_agent_tracker.update(agent, world)
                # Save to file
                best_agent_tracker.save_best_weights()
            
            # Close logger if enabled
            if agent_logger:
                agent_logger.close()
            
            return 0
          # TODO: Initialize agents and run simulation
        print("\n" + "="*60)
        print("SIMULATION READY")
        print("="*60)
        print("\nNext steps:")
        print("  1. Agent system implementation")
        print("  2. Simulation runner with generation management")
        print("  3. World update systems (growth, decay, etc.)")
        print("\nRun with --demo flag to see console visualization")
        print("Run with --gui flag to see Pygame GUI visualization")
        
        # Close logger if enabled
        if agent_logger:
            agent_logger.close()
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # Always shutdown pools before closing loggers to avoid background
        # tasks writing to closed file handles.
        try:
            from utils.parallel import shutdown_pool
            shutdown_pool()
        except:
            pass
        try:
            if Agent.logger is not None:
                Agent.logger.close()
                Agent.logger = None
            if Agent.world_model_logger is not None:
                Agent.world_model_logger.close()
                Agent.world_model_logger = None
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
