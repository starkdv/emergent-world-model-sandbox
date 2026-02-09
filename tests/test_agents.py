"""
Tests for the agent system.

Tests agent behavior, brain decision making, genome evolution,
and observation construction.

Author: Karan Vasa
Date: November 14, 2025
"""

import pytest
import numpy as np

from agents import (
    Agent, Brain, Genome, Action, ActionResult,
    create_default_trait_config, build_observation, get_observation_size
)
from world.world import World
from world.objects import WorldObject, EdibleComponent, SeedComponent


class TestGenome:
    """Test genome representation and genetic operations."""
    
    def test_random_genome_creation(self):
        """Test creating a random genome."""
        trait_config = create_default_trait_config()
        genome = Genome.random(
            weight_count=100,
            trait_config=trait_config,
            weight_init_std=0.5
        )
        
        assert len(genome.weights) == 100
        assert 'metabolism_rate' in genome.traits
        assert 'vision_radius' in genome.traits
        assert 0.5 <= genome.traits['metabolism_rate'] <= 2.0
        assert 2.0 <= genome.traits['vision_radius'] <= 10.0
    
    def test_uniform_crossover(self):
        """Test uniform crossover of two genomes."""
        trait_config = create_default_trait_config()
        parent_a = Genome.random(100, trait_config)
        parent_b = Genome.random(100, trait_config)
        
        child = Genome.mate(parent_a, parent_b, crossover_method='uniform')
        
        assert len(child.weights) == 100
        assert child.generation == max(parent_a.generation, parent_b.generation) + 1
        assert child.parent_ids == (parent_a.lineage_id, parent_b.lineage_id)
    
    def test_one_point_crossover(self):
        """Test one-point crossover."""
        trait_config = create_default_trait_config()
        parent_a = Genome.random(100, trait_config)
        parent_b = Genome.random(100, trait_config)
        
        child = Genome.mate(parent_a, parent_b, crossover_method='one_point')
        
        assert len(child.weights) == 100
    
    def test_blend_crossover(self):
        """Test blend crossover."""
        trait_config = create_default_trait_config()
        parent_a = Genome.random(100, trait_config)
        parent_b = Genome.random(100, trait_config)
        
        child = Genome.mate(parent_a, parent_b, crossover_method='blend')
        
        assert len(child.weights) == 100
    
    def test_mutation(self):
        """Test mutation changes weights."""
        trait_config = create_default_trait_config()
        parent = Genome.random(100, trait_config)
        
        # High mutation rate for testing
        child = Genome.mate(
            parent, parent,
            mutation_rate=1.0,  # 100% mutation
            mutation_std=0.5
        )
        
        # Should be different due to mutations
        assert not np.allclose(child.weights, parent.weights)
    
    def test_genome_copy(self):
        """Test genome copying."""
        trait_config = create_default_trait_config()
        original = Genome.random(100, trait_config)
        copy = original.copy()
        
        assert np.array_equal(copy.weights, original.weights)
        assert copy.traits == original.traits
        assert copy.lineage_id == original.lineage_id


class TestBrain:
    """Test neural network brain."""
    
    def test_brain_initialization(self):
        """Test brain is correctly initialized from genome."""
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count(64, [32, 16], 8)
        genome = Genome.random(weight_count, trait_config)
        
        brain = Brain(genome, input_size=64, hidden_sizes=[32, 16], output_size=8)
        
        assert len(brain.weights) == 3  # Input->H1, H1->H2, H2->Output
        assert len(brain.biases) == 3
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count(64, [32, 16], 8)
        genome = Genome.random(weight_count, trait_config)
        brain = Brain(genome)
        
        obs = np.random.randn(64)
        probs = brain.forward(obs)
        
        assert len(probs) == 8
        assert np.isclose(np.sum(probs), 1.0)  # Probabilities sum to 1
        assert np.all(probs >= 0) and np.all(probs <= 1)  # Valid probabilities
    
    def test_decide_action(self):
        """Test action decision from observation."""
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count()
        genome = Genome.random(weight_count, trait_config)
        brain = Brain(genome)
        
        obs = np.random.randn(64)
        action = brain.decide(obs)
        
        assert isinstance(action, Action)
        assert 0 <= action.value < 8
    
    def test_weight_count_calculation(self):
        """Test weight count calculation."""
        # Network: 64 -> 32 -> 16 -> 8
        # Weights: 64*32 + 32*16 + 16*8 = 2048 + 512 + 128 = 2688
        # Biases: 32 + 16 + 8 = 56
        # Total: 2744
        count = Brain.calculate_weight_count(64, [32, 16], 8)
        assert count == 2744


class TestAgent:
    """Test agent behavior."""
    
    @pytest.fixture
    def world(self):
        """Create a test world."""
        return World(width=20, height=20, seed=42)
    
    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count()
        genome = Genome.random(weight_count, trait_config)        
        return Agent(x=10, y=10, genome=genome, max_energy=100.0)
    
    def test_agent_creation(self, agent):
        """Test agent is correctly initialized."""
        assert agent.x == 10
        assert agent.y == 10
        assert agent.alive is True
        assert agent.energy == 100.0
        assert agent.age == 0
        assert len(agent.inventory) == 0
        assert agent.direction == (0, -1)  # North
    
    def test_agent_move_forward(self, world, agent):
        """Test agent can move forward."""
        world.add_agent(agent)
        
        # Find a passable tile and place agent there facing another passable tile
        from world.tiles import TerrainType
        for y in range(1, world.height - 1):
            for x in range(1, world.width - 1):
                current = world.tiles[y][x]
                north = world.tiles[y - 1][x]
                
                if current.is_passable() and north.is_passable():
                    agent.x = x
                    agent.y = y
                    agent.direction = (0, -1)  # Face north
                    initial_y = agent.y
                    
                    result = agent.execute_action(Action.MOVE_FORWARD, world)
                    
                    assert result.success
                    assert agent.y == initial_y - 1  # Moved north
                    return
        
        pytest.skip("No suitable tiles found for movement test")
    
    def test_agent_move_blocked(self, world, agent):
        """Test agent cannot move into rock."""
        world.add_agent(agent)
        
        # Find a rock tile with a passable tile next to it
        for y in range(world.height):
            for x in range(world.width):
                from world.tiles import TerrainType
                tile = world.tiles[y][x]
                if tile.terrain_type == TerrainType.ROCK:
                    # Check north neighbor
                    if y > 0 and world.tiles[y - 1][x].is_passable():
                        agent.x = x
                        agent.y = y - 1
                        agent.direction = (0, 1)  # Face south (towards rock)
                        
                        result = agent.execute_action(Action.MOVE_FORWARD, world)
                        assert not result.success
                        return
        
        # If no rocks found, skip test
        pytest.skip("No rock tiles in world")
    
    def test_agent_turn_left(self, agent, world):
        """Test agent can turn left."""
        world.add_agent(agent)
        agent.direction = (0, -1)  # North
        
        result = agent.execute_action(Action.TURN_LEFT, world)
        
        assert result.success
        assert agent.direction == (-1, 0)  # West
    
    def test_agent_turn_right(self, agent, world):
        """Test agent can turn right."""
        world.add_agent(agent)
        agent.direction = (0, -1)  # North
        
        result = agent.execute_action(Action.TURN_RIGHT, world)
        
        assert result.success
        assert agent.direction == (1, 0)  # East
    
    def test_agent_pick_up_object(self, world, agent):
        """Test agent can pick up objects."""
        world.add_agent(agent)
        
        # Create a berry on agent's tile
        berry = WorldObject(agent.x, agent.y)
        berry.add_component(EdibleComponent(calories=20.0))
        world.add_object(berry)
        
        result = agent.execute_action(Action.PICK_UP, world)
        
        assert result.success
        assert len(agent.inventory) == 1
        assert berry.id in agent.inventory
    
    def test_agent_drop_object(self, world, agent):
        """Test agent can drop objects."""
        world.add_agent(agent)
        
        # Give agent an object
        berry = WorldObject(agent.x, agent.y)
        berry.add_component(EdibleComponent(calories=20.0))
        world.add_object(berry)
        agent.inventory.append(berry.id)
        
        result = agent.execute_action(Action.DROP, world)
        
        assert result.success
        assert len(agent.inventory) == 0
        assert berry.id in world.tiles[agent.y][agent.x].object_ids
    
    def test_agent_eat_food(self, world, agent):
        """Test agent can eat food."""
        world.add_agent(agent)
        agent.energy = 50.0  # Set to half energy
        
        # Give agent a berry
        berry = WorldObject(agent.x, agent.y)
        berry.add_component(EdibleComponent(calories=20.0))
        world.add_object(berry)
        agent.inventory.append(berry.id)
        
        result = agent.execute_action(Action.EAT, world)
        
        assert result.success
        assert agent.energy > 50.0  # Energy increased
        assert len(agent.inventory) == 0  # Berry consumed
        assert berry.id not in world.objects  # Berry removed from world
    
    def test_agent_plant_seed(self, world, agent):
        """Test agent can plant seeds."""
        world.add_agent(agent)
        
        # Give agent a seed
        seed = WorldObject(agent.x, agent.y)
        seed.add_component(SeedComponent(
            plant_type="berry_plant",
            grow_time=50,
            required_fertility=0.3,
            required_moisture=0.2
        ))
        world.add_object(seed)
        agent.inventory.append(seed.id)
        
        # Ensure tile is plantable
        from world.tiles import TerrainType
        world.tiles[agent.y][agent.x].terrain_type = TerrainType.SOIL
        world.tiles[agent.y][agent.x].fertility = 0.5
        world.tiles[agent.y][agent.x].moisture = 0.5
        
        result = agent.execute_action(Action.USE, world)
        
        assert result.success
        assert len(agent.inventory) == 0  # Seed removed from inventory
        assert seed.id in world.tiles[agent.y][agent.x].object_ids  # Seed on ground
    
    def test_agent_metabolism(self, world, agent):
        """Test agent loses energy over time."""
        world.add_agent(agent)
        initial_energy = agent.energy
        
        agent.update(world)
        
        assert agent.energy < initial_energy
    
    def test_agent_dies_from_starvation(self, world, agent):
        """Test agent dies when energy reaches zero."""
        world.add_agent(agent)
        agent.energy = 0.1  # Very low energy
        
        agent.update(world)
        
        assert not agent.alive
    
    def test_agent_dies_from_old_age(self, world, agent):
        """Test agent dies from old age."""
        world.add_agent(agent)
        agent.age = agent.max_age
        
        agent.update(world)
        
        assert not agent.alive
    
    def test_agent_can_reproduce(self, agent):
        """Test reproduction requirements."""
        agent.energy = agent.max_energy * 0.8
        agent.age = 150
        
        assert agent.can_reproduce()
        
        agent.energy = agent.max_energy * 0.5  # Too low
        assert not agent.can_reproduce()


class TestObservation:
    """Test observation vector construction."""
    
    @pytest.fixture
    def world(self):
        """Create a test world."""
        return World(width=20, height=20, seed=42)
    
    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count()
        genome = Genome.random(weight_count, trait_config)
        return Agent(x=10, y=10, genome=genome)
    
    def test_observation_size(self):
        """Test observation vector has correct size."""
        size = get_observation_size()
        assert size == 64
    
    def test_build_observation(self, world, agent):
        """Test building observation vector."""
        world.add_agent(agent)
        obs = build_observation(agent, world)
        
        assert len(obs) == 64
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))  # No NaN or Inf
    
    def test_observation_includes_agent_state(self, world, agent):
        """Test observation includes agent state."""
        world.add_agent(agent)
        agent.energy = 50.0
        agent.max_energy = 100.0
        
        obs = build_observation(agent, world)
        
        # First feature should be energy ratio
        assert np.isclose(obs[0], 0.5)
    
    def test_observation_includes_vision(self, world, agent):
        """Test observation includes vision grid."""
        world.add_agent(agent)
        
        # Place a berry in agent's vision
        berry = WorldObject(agent.x + 1, agent.y)
        berry.add_component(EdibleComponent(calories=20.0))
        world.add_object(berry)
        
        obs = build_observation(agent, world)
        
        # Vision features should be non-zero
        assert np.any(obs[8:58] != 0)  # Vision is features 8-57
    
    def test_observation_normalized(self, world, agent):
        """Test observation values are normalized."""
        world.add_agent(agent)
        obs = build_observation(agent, world)
        
        # Most values should be in [0, 1] range
        # (some can be slightly outside due to encoding choices)
        assert np.all(obs >= -1.1)
        assert np.all(obs <= 1.1)


class TestAgentIntegration:
    """Integration tests for agents in world."""
    
    def test_agents_update_in_world(self):
        """Test agents are updated each tick."""
        world = World(width=20, height=20, seed=42)
        
        # Create and add agents
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count()
        
        for i in range(5):
            genome = Genome.random(weight_count, trait_config)
            agent = Agent(x=5 + i, y=5, genome=genome)
            world.add_agent(agent)
        
        assert len(world.agents) == 5
        
        # Update world
        world.update()
        
        # All agents should have aged
        for agent in world.agents.values():
            assert agent.age == 1
    
    def test_dead_agents_removed(self):
        """Test dead agents are removed from world."""
        world = World(width=20, height=20, seed=42)
        
        trait_config = create_default_trait_config()
        weight_count = Brain.calculate_weight_count()
        genome = Genome.random(weight_count, trait_config)
        agent = Agent(x=10, y=10, genome=genome)
        agent.energy = 0.1  # Very low energy
        world.add_agent(agent)
        
        # Update until agent dies
        world.update()
        
        # Agent should be removed
        assert len(world.agents) == 0
