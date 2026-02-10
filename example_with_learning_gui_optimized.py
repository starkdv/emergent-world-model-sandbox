"""
Example: High-performance GUI with async logging and multi-core optimization.

Optimizations:
- Async logging with batched I/O (no blocking writes)
- Performance monitoring (FPS, update time, render time)
- Configurable target FPS
- Logger statistics display

Usage:
    python example_with_learning_gui_optimized.py

Controls:
    - Click on agents to select them
    - See learning stats in real-time
    - ESC to exit
    - F key to show FPS stats

Author: Karan Vasa
Date: February 10, 2026
"""

import random
import pygame
import time
from world.world import World
from world.objects import WorldObject, EdibleComponent, SeedComponent, PlantComponent
from agents import Agent, Genome, create_default_trait_config
from utils.ui.pygame_renderer import PygameRenderer
from utils.data.async_logger import AsyncWorldModelLogger


def populate_world_with_resources(world, berry_count=30, plant_count=20, seed_count=15):
    """Add initial resources to the world."""
    # Add berries
    for _ in range(berry_count):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        if world.is_valid_position(x, y):
            berry = WorldObject(x, y)
            berry.add_component(EdibleComponent(calories=25.0))
            world.add_object(berry)
    
    # Add plants
    for _ in range(plant_count):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        tile = world.get_tile(x, y)
        if tile and tile.is_plantable():
            plant = WorldObject(x, y)
            plant.add_component(PlantComponent(
                mature_age=30,
                max_age=200,
                spawn_rate=0.1
            ))
            world.add_object(plant)
    
    # Add seeds
    for _ in range(seed_count):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        if world.is_valid_position(x, y):
            seed = WorldObject(x, y)
            seed.add_component(SeedComponent(
                plant_type="berry_plant",
                grow_time=50,
                max_age=200
            ))
            world.add_object(seed)


def create_learning_agent(x, y):
    """Create an agent with learning enabled."""
    trait_config = create_default_trait_config()
    genome = Genome.random(weight_count=2744, trait_config=trait_config)
    agent = Agent(x=x, y=y, genome=genome)
    
    # Enable learning
    agent.enable_learning(
        learning_rate=0.01,
        discount_factor=0.95,
        batch_size=16,
        buffer_capacity=1000
    )
    
    return agent


def main():
    """Run simulation with high-performance logging and rendering."""
    print("=" * 70)
    print("HIGH-PERFORMANCE SIMULATION - OPTIMIZED GUI")
    print("=" * 70)
    
    # Configuration
    WORLD_SIZE = 40
    NUM_AGENTS = 15
    TILE_SIZE = 16
    TARGET_FPS = 60  # Increased from 30
    ENABLE_LOGGING = True  # Set to False to disable logging entirely
    
    # Create world
    print(f"\nInitializing {WORLD_SIZE}x{WORLD_SIZE} world...")
    world = World(width=WORLD_SIZE, height=WORLD_SIZE)
    populate_world_with_resources(world, berry_count=60, plant_count=40, seed_count=30)
    
    # Create agents with learning
    print(f"Creating {NUM_AGENTS} learning-enabled agents...")
    agents = []
    for i in range(NUM_AGENTS):
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        agent = create_learning_agent(x, y)
        world.add_agent(agent)
        agents.append(agent)
        print(f"  Agent {i}: pos=({x},{y}), learning=ON")
    
    # Initialize async logger
    logger = None
    if ENABLE_LOGGING:
        print(f"\nInitializing async logger (batch_size=100)...")
        logger = AsyncWorldModelLogger(
            output_dir="data/logs",
            log_every_n_ticks=10,  # Log world state every 10 ticks
            batch_size=100,  # Batch 100 entries before writing
            flush_interval=2.0,  # Force flush every 2 seconds
            queue_maxsize=10000  # Large queue to prevent blocking
        )
        Agent.world_model_logger = logger
    else:
        print(f"\nLogging DISABLED")
    
    # Create renderer
    print(f"\nStarting GUI renderer (target: {TARGET_FPS} FPS)...")
    renderer = PygameRenderer(world, tile_size=TILE_SIZE, target_fps=TARGET_FPS)
    
    print("\n" + "=" * 70)
    print("GUI CONTROLS:")
    print("  • Click on agents to see their learning stats")
    print("  • F key to toggle FPS display")
    print("  • ESC to exit")
    print("  • Watch as agents learn to survive!")
    print("=" * 70 + "\n")
    
    # Performance tracking
    clock = pygame.time.Clock()
    frame_times = []
    update_times = []
    render_times = []
    show_fps = False
    
    # Main loop
    running = True
    tick = 0
    
    try:
        while running:
            frame_start = time.perf_counter()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_f:
                        show_fps = not show_fps
            
            # Update simulation
            update_start = time.perf_counter()
            world.update()
            
            # Log world state (async, non-blocking)
            if logger:
                logger.log_world_state(tick, world)
            
            tick += 1
            update_time = time.perf_counter() - update_start
            update_times.append(update_time * 1000)  # Convert to ms
            
            # Render
            render_start = time.perf_counter()
            renderer.render()
            
            # Draw FPS overlay if enabled
            if show_fps:
                draw_performance_overlay(
                    renderer.screen,
                    clock.get_fps(),
                    update_times,
                    render_times,
                    logger.get_stats() if logger else None
                )
            
            pygame.display.flip()
            render_time = time.perf_counter() - render_start
            render_times.append(render_time * 1000)  # Convert to ms
            
            # Limit lists to last 60 samples
            if len(update_times) > 60:
                update_times.pop(0)
            if len(render_times) > 60:
                render_times.pop(0)
            
            # Cap frame rate
            clock.tick(TARGET_FPS)
            
            frame_time = time.perf_counter() - frame_start
            frame_times.append(frame_time * 1000)
            if len(frame_times) > 60:
                frame_times.pop(0)
            
            # Check if all agents dead
            alive_count = sum(1 for a in agents if a.alive)
            if alive_count == 0:
                print(f"\nAll agents died at tick {tick}")
                pygame.time.wait(3000)
                break
            
            # Print periodic stats
            if tick % 100 == 0:
                alive_agents = [a for a in agents if a.alive]
                if alive_agents:
                    avg_energy = sum(a.energy for a in alive_agents) / len(alive_agents)
                    avg_experiences = sum(len(a.learner.replay_buffer) for a in alive_agents) / len(alive_agents)
                    avg_fps = sum(frame_times) / len(frame_times) if frame_times else 0
                    avg_update = sum(update_times) / len(update_times) if update_times else 0
                    print(f"Tick {tick:4d}: {len(alive_agents):2d} alive | "
                          f"Energy: {avg_energy:5.1f} | "
                          f"Exp: {avg_experiences:5.1f} | "
                          f"Frame: {avg_fps:5.2f}ms | "
                          f"Update: {avg_update:4.2f}ms")
                    
                    if logger:
                        stats = logger.get_stats()
                        print(f"          Logger: {stats['transitions_logged']} trans, "
                              f"{stats['batches_written']} batches, "
                              f"Q: {stats['queue_size']}")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        renderer.close()
        
        if logger:
            logger.close()
    
    # Final report
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    
    alive_agents = [a for a in agents if a.alive]
    dead_agents = [a for a in agents if not a.alive]
    
    print(f"\nFinal Statistics:")
    print(f"  Total ticks: {tick}")
    print(f"  Survivors: {len(alive_agents)}/{NUM_AGENTS}")
    print(f"  Deaths: {len(dead_agents)}")
    
    if frame_times:
        avg_frame = sum(frame_times) / len(frame_times)
        avg_fps = 1000.0 / avg_frame if avg_frame > 0 else 0
        print(f"\nPerformance:")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Average frame time: {avg_frame:.2f}ms")
        if update_times:
            print(f"  Average update time: {sum(update_times)/len(update_times):.2f}ms")
        if render_times:
            print(f"  Average render time: {sum(render_times)/len(render_times):.2f}ms")
    
    if alive_agents:
        print(f"\nTop 5 Survivors:")
        for i, agent in enumerate(sorted(alive_agents, key=lambda a: a.fitness, reverse=True)[:5], 1):
            exp_count = len(agent.learner.replay_buffer) if agent.learner else 0
            print(f"  {i}. Agent {agent.id}: "
                  f"Age={agent.age:3d}, "
                  f"Energy={agent.energy:5.1f}, "
                  f"Fitness={agent.fitness:6.1f}, "
                  f"Exp={exp_count:4d}")
        
        # Best agent
        best = max(alive_agents, key=lambda a: a.fitness)
        print(f"\n🏆 Best Agent: #{best.id}")
        print(f"   Fitness: {best.fitness:.1f}")
        print(f"   Age: {best.age}")
        print(f"   Energy: {best.energy:.1f}")
        print(f"   Experiences: {len(best.learner.replay_buffer)}")
    
    if dead_agents:
        avg_death_age = sum(a.age for a in dead_agents) / len(dead_agents)
        print(f"\nDead agents average age: {avg_death_age:.1f}")
    
    print("\n" + "=" * 70)
    print("HIGH-PERFORMANCE SIMULATION: ✅ COMPLETE!")
    print("=" * 70)


def draw_performance_overlay(screen, fps, update_times, render_times, logger_stats):
    """Draw performance metrics overlay."""
    font = pygame.font.Font(None, 24)
    y_offset = 10
    
    # Background panel
    panel_width = 280
    panel_height = 160 if logger_stats else 120
    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel.fill((35, 35, 45, 230))
    screen.blit(panel, (screen.get_width() - panel_width - 10, y_offset))
    
    # Text
    x_pos = screen.get_width() - panel_width
    y_offset += 10
    
    # FPS
    fps_text = font.render(f"FPS: {fps:.1f}", True, (100, 255, 100))
    screen.blit(fps_text, (x_pos, y_offset))
    y_offset += 25
    
    # Update time
    if update_times:
        avg_update = sum(update_times) / len(update_times)
        update_text = font.render(f"Update: {avg_update:.2f}ms", True, (255, 255, 100))
        screen.blit(update_text, (x_pos, y_offset))
        y_offset += 25
    
    # Render time
    if render_times:
        avg_render = sum(render_times) / len(render_times)
        render_text = font.render(f"Render: {avg_render:.2f}ms", True, (100, 200, 255))
        screen.blit(render_text, (x_pos, y_offset))
        y_offset += 25
    
    # Frame time
    if update_times and render_times:
        avg_frame = (sum(update_times) + sum(render_times)) / len(update_times)
        frame_text = font.render(f"Frame: {avg_frame:.2f}ms", True, (255, 200, 100))
        screen.blit(frame_text, (x_pos, y_offset))
        y_offset += 25
    
    # Logger stats
    if logger_stats:
        queue_size = logger_stats.get('queue_size', 0)
        batches = logger_stats.get('batches_written', 0)
        logger_text = font.render(f"Log Q: {queue_size} | B: {batches}", True, (200, 150, 255))
        screen.blit(logger_text, (x_pos, y_offset))


if __name__ == "__main__":
    main()
