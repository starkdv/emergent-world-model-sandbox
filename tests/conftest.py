import sys
import os
import pytest

# Add the project root to sys.path so that tests can import modules
# This assumes the tests directory is directly under the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Basic fixture for World if needed globally
@pytest.fixture
def clean_world():
    from world.world import World
    return World(width=10, height=10)
