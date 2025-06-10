#!/usr/bin/env python3
"""
Parameter space generation utility for throughput experiments.

This script generates parameter configurations for UQ methods based on the 
parameter spaces defined in the configuration file. It supports:
- Regular grid sampling for continuous parameters
- Exhaustive sampling for discrete parameters
- Log-scale sampling where specified
- Automatic handling of parameter bounds and fixed values

Usage:
    python generate_parameter_grid.py --config config.yaml --num_samples 100 --output parameter_configs.json
"""

import click
import yaml
import json
import numpy as np
import itertools
from typing import Dict, List, Any, Tuple
from pathlib import Path





def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file {config_path} not found")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        exit(1)


def generate_parameter_values(param_config: Dict[str, Any], num_points: int) -> List[Any]:
    """Generate parameter values based on parameter configuration."""
    
    param_type = param_config['type']
    
    if param_type == 'fixed':
        # Fixed parameter - return single value
        return [param_config['value']]
    
    elif param_type == 'range':
        # Range parameter - generate evenly spaced values
        bounds = param_config['bounds']
        min_val, max_val = bounds[0], bounds[1]
        log_scale = param_config.get('log_scale', False)
        
        # For integer ranges, cap num_points to avoid duplicates
        if isinstance(min_val, int) and isinstance(max_val, int):
            max_unique_values = max_val - min_val + 1
            num_points = min(num_points, max_unique_values)
        
        if log_scale:
            values = np.logspace(np.log10(min_val), np.log10(max_val), num_points)
        else:
            # Linear sampling
            values = np.linspace(min_val, max_val, num_points)
        
        # Convert to appropriate type if needed
        if isinstance(min_val, int) and isinstance(max_val, int):
            values = np.round(values).astype(int)
            values = sorted(set(map(int, values)))
            return values
        
        return values
    
    elif param_type == 'choice':
        return param_config['values']
    
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def generate_method_configurations(method_name: str, method_config: Dict[str, Any], 
                                 target_samples: int) -> List[Dict[str, Any]]:
    """Generate all parameter configurations for a specific UQ method."""
    
    parameter_space = method_config.get('parameter_space', [])
    
    if not parameter_space:
        # No parameters for this method (e.g., no_uq)
        return [{'uq_method': method_name, 'parameters': {}}]
    
    # Generate parameter values for each parameter
    param_values = {}
    param_names = []
    
    for param_config in parameter_space:
        param_name = param_config['name']
        param_names.append(param_name)
        
        # Determine number of points for this parameter
        if param_config['type'] == 'fixed':
            num_points = 1
        elif param_config['type'] == 'choice':
            num_points = len(param_config['values'])
        else:
            # For range parameters, we'll calculate the grid size dynamically
            num_points = None
        
        param_values[param_name] = (param_config, num_points)
    
    # Calculate grid dimensions
    if len(param_names) == 0:
        return [{'uq_method': method_name, 'parameters': {}}]
    
    # Calculate optimal grid size for each parameter
    total_params = len([p for p in parameter_space if p['type'] != 'fixed'])
    if total_params == 0:
        # All parameters are fixed
        configs = []
        param_dict = {}
        for param_config in parameter_space:
            param_dict[param_config['name']] = param_config['value']
        configs.append({'uq_method': method_name, 'parameters': param_dict})
        return configs
    
    # For multiple parameters, calculate grid size per dimension
    if total_params == 1:
        points_per_param = target_samples
    elif total_params == 2:
        points_per_param = int(np.sqrt(target_samples))
    else:
        points_per_param = int(target_samples ** (1.0 / total_params))
    
    # Generate values for each parameter
    param_value_lists = []
    for param_name in param_names:
        param_config, fixed_num_points = param_values[param_name]
        
        if fixed_num_points is not None:
            # Fixed number of points (fixed or choice parameters)
            values = generate_parameter_values(param_config, fixed_num_points)
        else:
            # Range parameter - use calculated grid size
            values = generate_parameter_values(param_config, points_per_param)
        
        param_value_lists.append(values)
    
    # Generate all combinations
    configs = []
    for param_combination in itertools.product(*param_value_lists):
        param_dict = {}
        for i, param_name in enumerate(param_names):
            param_dict[param_name] = param_combination[i]
        
        configs.append({
            'uq_method': method_name,
            'parameters': param_dict
        })
    
    # If we have too many configurations, sample them
    if len(configs) > target_samples:
        indices = np.linspace(0, len(configs) - 1, target_samples, dtype=int)
        configs = [configs[i] for i in indices]
    
    return configs


def generate_all_configurations(config: Dict[str, Any], target_samples: int, 
                               selected_methods: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Generate configurations for all UQ methods."""
    
    uq_methods = config.get('uq_methods', {})
    all_configs = {}
    
    for method_name, method_config in uq_methods.items():
        # Skip if specific methods were requested and this isn't one of them
        if selected_methods and method_name not in selected_methods:
            continue
        
        print(f"Generating configurations for {method_name}...")
        
        try:
            method_configs = generate_method_configurations(method_name, method_config, target_samples)
            all_configs[method_name] = method_configs
            print(f"  Generated {len(method_configs)} configurations")
        except Exception as e:
            print(f"  Error generating configurations for {method_name}: {e}")
            all_configs[method_name] = []
    
    # Always include no_uq baseline if not explicitly excluded
    if not selected_methods or 'no_uq' in selected_methods:
        if 'no_uq' not in all_configs:
            all_configs['no_uq'] = [{'uq_method': 'no_uq', 'parameters': {}}]
            print("Added no_uq baseline configuration")
    
    return all_configs


def save_configurations(all_configs: Dict[str, List[Dict[str, Any]]], output_path: str):
    """Save parameter configurations to JSON file."""
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten configurations into single list with metadata
        flattened_configs = []
        for method_name, configs in all_configs.items():
            for i, config in enumerate(configs):
                flattened_config = {
                    'config_id': f"{method_name}_{i:03d}",
                    'method': method_name,
                    'uq_method': config['uq_method'],
                    'parameters': config['parameters']
                }
                flattened_configs.append(flattened_config)
        
        # Save metadata and configurations
        output_data = {
            'metadata': {
                'total_configurations': len(flattened_configs),
                'methods': list(all_configs.keys()),
                'configs_per_method': {method: len(configs) for method, configs in all_configs.items()}
            },
            'configurations': flattened_configs
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nConfigurations saved to {output_path}")
        print(f"Total configurations: {len(flattened_configs)}")
        
    except Exception as e:
        print(f"Error saving configurations: {e}")
        raise


def print_summary(all_configs: Dict[str, List[Dict[str, Any]]]):
    """Print summary of generated configurations."""
    print(f"\n{'='*60}")
    print("PARAMETER GENERATION SUMMARY")
    print(f"{'='*60}")
    
    total_configs = 0
    for method_name, configs in all_configs.items():
        print(f"{method_name}: {len(configs)} configurations")
        if configs and 'parameters' in configs[0]:
            # Show parameter ranges
            params = configs[0]['parameters']
            if params:
                print(f"  Parameters: {list(params.keys())}")
                # Show value ranges for first few configs
                if len(configs) > 1:
                    for param_name in params.keys():
                        values = [cfg['parameters'][param_name] for cfg in configs[:5]]
                        if len(set(values)) > 1:
                            print(f"    {param_name}: {min(values):.4g} to {max(values):.4g} (sample)")
        total_configs += len(configs)
    
    print(f"\nTotal configurations: {total_configs}")
    print(f"{'='*60}")


@click.command()
@click.option('--config', default='config.yaml', help='Configuration file path')
@click.option('--num-samples', default=100, help='Target number of parameter samples per method')
@click.option('--output', default='parameter_configs.json', help='Output file path for parameter configurations')
@click.option('--methods', multiple=True, help='Specific UQ methods to generate configs for (default: all)')
def main(config, num_samples, output, methods):
    """Generate parameter grid for UQ methods."""
    
    print(f"Loading configuration from {config}")
    config_data = load_config(config)
    
    print(f"Generating parameter configurations (target: {num_samples} per method)")
    
    # Generate configurations
    methods_list = list(methods) if methods else None
    all_configs = generate_all_configurations(config_data, num_samples, methods_list)
    
    # Print summary
    print_summary(all_configs)
    
    # Save configurations
    save_configurations(all_configs, output)
    
    print("\nParameter generation completed successfully!")


if __name__ == "__main__":
    main() 