"""
Example: Running the simulation with learning enabled - GUI VERSION.

This demonstrates learning agents with visual feedback.

Usage:
    python example_with_learning_gui.py

Controls:
    - Click on agents to select them
    - See learning stats in real-time
    - ESC to exit

Author: Karan Vasa
Date: November 15, 2025
"""

import random
import pygame
from world.world import World
from world.objects import WorldObject, EdibleComponent, SeedComponent, PlantComponent
from agents import Agent, Genome, create_default_trait_config
from utils.ui.pygame_renderer import PygameRenderer


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
        learning_rate=0.01,       # Learning speed
        discount_factor=0.95,     # Future reward importance
        batch_size=16,            # Experiences per update
        buffer_capacity=1000      # Max experiences stored
    )
    
    return agent


def main():
    """Run simulation with learning-enabled agents in GUI."""
    print("=" * 70)
    print("SIMULATION WITH LEARNING - GUI MODE")
    print("=" * 70)
    
    # Configuration
    WORLD_SIZE = 40
    NUM_AGENTS = 15
    TILE_SIZE = 16
    
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
    
    # Create renderer
    print(f"\nStarting GUI renderer...")
    renderer = PygameRenderer(world, tile_size=TILE_SIZE, target_fps=30)
    
    print("\n" + "=" * 70)
    print("GUI CONTROLS:")
    print("  • Click on agents to see their learning stats")
    print("  • ESC to exit")
    print("  • Watch as agents learn to survive!")
    print("=" * 70 + "\n")
    
    # Main loop
    running = True
    tick = 0
    clock = pygame.time.Clock()
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Update simulation
        world.update()
        tick += 1
        
        # Render
        renderer.render()
        clock.tick(30)  # 30 FPS
        
        # Check if all agents dead
        alive_count = sum(1 for a in agents if a.alive)
        if alive_count == 0:
            print(f"\nAll agents died at tick {tick}")
            # Keep window open for a bit
            pygame.time.wait(3000)
            break
        
        # Print periodic stats
        if tick % 100 == 0:
            alive_agents = [a for a in agents if a.alive]
            if alive_agents:
                avg_energy = sum(a.energy for a in alive_agents) / len(alive_agents)
                avg_experiences = sum(len(a.learner.replay_buffer) for a in alive_agents) / len(alive_agents)
                print(f"Tick {tick:4d}: {len(alive_agents):2d} alive | "
                      f"Avg Energy: {avg_energy:5.1f} | "
                      f"Avg Experiences: {avg_experiences:5.1f}")
    
    # Cleanup
    renderer.close()
    
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
    print("LEARNING SYSTEM: ✅ VISUALIZED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
