# Emergent World-Model Sandbox - Development Guidelines

**Author:** Karan Vasa  
**Date:** November 14, 2025

## General Development Principles

### 1. File Organization
- **DO NOT create unnecessary .md files** unless specifically requested
- Keep documentation minimal and focused
- Use clear, descriptive file names
- Follow the specified module structure from the design document

### 2. Code Quality Standards

#### Documentation Requirements
- **ALL functions and classes MUST have proper docstrings** following Python conventions:
  ```python
  def function_name(param1: type, param2: type) -> return_type:
      """
      Brief description of function purpose.
      
      Args:
          param1: Description of parameter 1
          param2: Description of parameter 2
          
      Returns:
          Description of return value
          
      Raises:
          ExceptionType: Description of when this exception is raised
      """
  ```

- Use type hints for all function parameters and return values
- Add inline comments for complex algorithms or non-obvious logic
- Document class attributes and their purposes

#### Function Design
- Keep functions small and focused on a single responsibility
- Use descriptive function and variable names
- Avoid deep nesting (max 3 levels recommended)
- Use early returns to reduce complexity
- Handle edge cases explicitly

### 3. Architecture Guidelines

#### Modularity
- Maintain strict separation between world physics and agent logic
- Use dependency injection where appropriate
- Keep modules loosely coupled
- Design for testability from the start

#### Component System
- Follow Entity-Component-System (ECS) pattern for world objects
- Components should be pure data structures
- Systems should be stateless functions operating on components
- Avoid circular dependencies between components

#### Performance Considerations
- Use numpy arrays for large numerical computations
- Minimize object creation in simulation loops
- Cache expensive calculations when possible
- Profile before optimizing

### 4. Testing Standards

#### Unit Testing
- Write tests for all public functions and methods
- Use descriptive test names that explain what is being tested
- Follow AAA pattern (Arrange, Act, Assert)
- Mock external dependencies
- Aim for >90% code coverage

#### Integration Testing
- Test complete simulation runs
- Verify world state consistency after operations
- Test agent behavior under various conditions
- Validate evolutionary mechanics

### 5. Git and Version Control

#### Commit Messages
- Use conventional commit format: `type(scope): description`
- Types: feat, fix, docs, style, refactor, test, chore
- Keep commits atomic and focused
- Include issue numbers when relevant

#### Branch Strategy
- Use feature branches for new development
- Keep main branch stable and deployable
- Require code review for all changes
- Run tests before merging

### 6. Error Handling

#### Exception Management
- Use specific exception types rather than generic `Exception`
- Provide meaningful error messages
- Log errors with sufficient context for debugging
- Handle expected errors gracefully
- Let unexpected errors propagate

#### Input Validation
- Validate all public function inputs
- Use assertions for development-time checks
- Provide clear error messages for invalid inputs

### 7. Configuration Management

#### Settings Structure
- Use a centralized configuration system
- Make all simulation parameters configurable
- Provide sensible defaults
- Document all configuration options
- Support configuration via files and environment variables

#### Constants
- Define magic numbers as named constants
- Group related constants in appropriate modules
- Use UPPER_CASE naming for constants

### 8. Specific Guidelines for This Project

#### Emergent Behavior Focus
- **NEVER hardcode high-level behaviors** (farming, cooperation, etc.)
- Implement only primitive actions (MOVE, TURN, PICK_UP, DROP, EAT)
- Let complex behaviors emerge through evolution
- Avoid giving agents explicit knowledge about world mechanics

#### World Physics
- Keep world update systems deterministic
- Document all world rules clearly
- Make physics parameters configurable
- Ensure world state is always consistent

#### Agent Design
- Neural networks should be simple MLPs initially
- Genome representation should be flexible for future extensions
- Observation vectors should be normalized and bounded
- Action space should be continuous or discrete as appropriate

#### Evolution System
- Use well-established genetic algorithm techniques
- Make selection and mutation strategies configurable
- Track lineage and diversity metrics
- Avoid premature optimization of evolutionary parameters

### 9. Development Workflow

#### GitHub Copilot Usage
- Provide clear, specific prompts with context
- Review all AI-generated code carefully
- Test AI-generated functions thoroughly
- Refactor AI code to match project standards

#### Code Review Checklist
- [ ] All functions have proper docstrings
- [ ] Type hints are used consistently
- [ ] No hardcoded high-level behaviors
- [ ] Tests are included for new functionality
- [ ] Error handling is appropriate
- [ ] Performance implications are considered
- [ ] Configuration is properly externalized

### 10. Documentation Standards

#### Code Comments
- Explain WHY, not just WHAT
- Update comments when code changes
- Remove outdated comments
- Use TODO comments for known improvements

#### API Documentation
- Document all public interfaces
- Provide usage examples
- Document expected behavior and side effects
- Keep documentation synchronized with code

### 11. Security and Safety

#### Input Sanitization
- Validate all external inputs
- Prevent injection attacks in configuration
- Sanitize file paths and names

#### Resource Management
- Implement proper cleanup for resources
- Monitor memory usage in long-running simulations
- Set reasonable limits on simulation parameters

### 12. Performance Guidelines

#### Optimization Strategy
- Measure before optimizing
- Focus on algorithmic improvements first
- Use profiling tools to identify bottlenecks
- Document performance characteristics

#### Scalability Considerations
- Design for larger world sizes
- Consider parallelization opportunities
- Plan for distributed simulation capabilities
- Monitor resource usage patterns

### 13. UI and Visualization Guidelines

#### UI Framework Selection
- Use pygame for real-time 2D visualization
- Consider tkinter for simple control panels
- Implement matplotlib for data plotting and charts
- Keep UI code separate from simulation logic

#### Visualization Standards
- Use consistent color schemes for different entity types
- Implement zoom and pan functionality for large worlds
- Provide clear visual indicators for agent states
- Update display at configurable frame rates (30-60 FPS)
- Include performance metrics in the UI (FPS, simulation speed)

#### User Interface Design
- Keep controls intuitive and accessible
- Provide keyboard shortcuts for common actions
- Implement tooltips and help text
- Support different display resolutions
- Allow customization of visual elements

### 14. Data Collection and Analysis Guidelines

#### Data Logging Standards
- Use structured formats (CSV, JSON, Parquet) for data export
- Include timestamps and unique identifiers for all records
- Log at multiple granularities (per-tick, per-generation, per-event)
- Compress large data files automatically
- Implement data validation and integrity checks

#### CSV Export Format
```python
# Agent behavior CSV structure:
timestamp, agent_id, generation, x, y, energy, action, target_object_id, fitness, age
# Population CSV structure:
generation, population_size, avg_fitness, max_fitness, genetic_diversity, extinct_lineages
# World state CSV structure:
tick, total_objects, total_food, total_seeds, total_plants, avg_fertility
```

#### Data Privacy and Ethics
- Avoid collecting unnecessary personal information
- Document all data collection practices
- Provide options to disable data collection
- Respect user privacy in all analytics

#### Analysis Tools Integration
- Design for compatibility with pandas, numpy, and scipy
- Provide utility functions for common analyses
- Support statistical significance testing
- Include data visualization helper functions

## Project-Specific Conventions

### Naming Conventions
- Classes: `PascalCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_CASE`
- Private members: `_leading_underscore`
- Modules: `lowercase`

### File Structure
```
emergent_world_model/
├── world/
├── agents/
├── simulation/
├── utils/
│   ├── render.py
│   ├── ui/
│   │   ├── main_window.py
│   │   ├── world_view.py
│   │   └── control_panel.py
│   └── data/
│       ├── logger.py
│       ├── exporter.py
│       └── analyzer.py
├── tests/
├── config/
└── data/
    ├── exports/
    ├── logs/
    └── analysis/
```

### Import Organization
1. Standard library imports
2. Third-party imports
3. Local project imports
4. Separate groups with blank lines

## Final Notes

This project aims to create emergent behaviors through evolution, not to implement them directly. Keep this principle in mind for all development decisions. When in doubt, prefer simpler, more primitive implementations that allow for emergence rather than complex, directed solutions.

Remember: The goal is to observe fascinating emergent behaviors, not to program them explicitly.
