"""Quick test to verify max_population setting works."""
import yaml

# Load config
with open('config/training_easy.yaml') as f:
    config = yaml.safe_load(f)

# Check reproduction settings
repro = config.get('reproduction', {})
print('Reproduction Configuration:')
print(f'  enabled: {repro.get("enabled")}')
print(f'  max_population: {repro.get("max_population")}')

# Verify it's set correctly
max_pop = repro.get('max_population')
if max_pop == 50:
    print('\n✅ max_population is correctly set to 50')
else:
    print(f'\n❌ max_population is {max_pop}, expected 50')
