"""
Parsl workflow driver for model size vs throughput experiments.

This script runs throughput experiments for all parameter configurations generated
by generate_parameter_grid.py using Parsl for parallel execution on clusters or locally.

Usage:
    # First generate parameter configurations
    python generate_parameter_grid.py --config config.yaml --num-samples 100 --output parameter_configs.json
    
    # Then run the throughput workflow
    python size_vs_throughput_driver.py --param_config parameter_configs.json --config config.yaml --output results/
    
    # Use different model architectures
    python size_vs_throughput_driver.py --param_config parameter_configs.json --config config.yaml --output results/ --model_name resnet_50_size
    python size_vs_throughput_driver.py --param_config parameter_configs.json --config config.yaml --output results/ --model_name resnet_101_size
"""

import click
import os
import parsl
import yaml
import json
import pandas as pd
from pathlib import Path
from parsl.app.app import python_app
from parsl.app.app import bash_app
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.providers import LocalProvider
from parsl.launchers import SingleNodeLauncher
from parsl.data_provider.files import File as ParslFile
from parsl.executors import HighThroughputExecutor


def get_config(config_filename):
    """Load configuration from YAML file."""
    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_parameter_configs(param_config_path):
    """Load parameter configurations from JSON file."""
    with open(param_config_path, 'r') as f:
        param_data = json.load(f)
    return param_data


@bash_app(cache=True)
def run_throughput_experiment(config_file, param_config_file, config_id, dataset_name, model_name, output_file,
                              stdout=parsl.AUTO_LOGNAME,
                              stderr=parsl.AUTO_LOGNAME):
    """Run a single throughput experiment for a given configuration."""
    import sh
    import os
    
    # Clear SLURM environment variables to avoid conflicts
    try:
        os.unsetenv('SLURM_CPU_BIND')
        os.unsetenv('SLURM_CPU_BIND_LIST')
        os.unsetenv('SLURM_CPUS_ON_NODE')
        os.unsetenv('SLURM_CPUS_PER_TASK')
        os.unsetenv('SLURM_CPU_BIND_TYPE')
        os.unsetenv('SLURM_JOB_NAME')
    except KeyError:
        pass
    
    python = sh.Command('python3')
    
    # Build the command arguments
    args = ['size_vs_throughput.py',
            '--config_id', config_id,
            '--param_config', param_config_file,
            '--config', config_file,
            '--output', output_file,
            '--dataset_name', dataset_name,
            '--model_name', model_name]
    
    command = python.bake(*args)
    print(f"Running command: {str(command)}")
    return str(command)


@python_app
def combine_throughput_results(result_files, output_file):
    """Combine all throughput experiment results into a single CSV file."""
    import pandas as pd
    import json
    import os
    from pathlib import Path
    
    combined_results = []
    
    for file_path in result_files:
        try:
            if os.path.exists(file_path):
                # Load JSON result file
                with open(file_path, 'r') as f:
                    result_data = json.load(f)
                
                # Convert to flat dictionary for CSV
                metric_results = result_data.get('metric_results', {})
                flat_result = {
                    'config_id': os.path.basename(file_path).replace('.json', ''),
                    'uq_method': result_data.get('uq_method', 'UNKNOWN'),
                    'total_params': result_data.get('total_params', 0),
                    'trainable_params': result_data.get('trainable_params', 0),
                    'model_size_mb': result_data.get('model_size_mb', 0.0),
                    'uncertainty_estimating_throughput': metric_results.get('uncertainty_estimating_throughput', 0.0),
                    'throughput_std': metric_results.get('throughput_std', 0.0),
                    'max_memory_usage': metric_results.get('max_memory_usage', 0.0),
                    'experiment_timestamp': result_data.get('experiment_timestamp', ''),
                    'base_architecture': result_data.get('base_architecture', ''),
                    'dataset_name': result_data.get('dataset_name', ''),
                }
                
                # Add parameter values as separate columns
                parameters = result_data.get('parameters', {})
                for param_name, param_value in parameters.items():
                    flat_result[f'param_{param_name}'] = param_value
                
                combined_results.append(flat_result)
            else:
                # Create a record for failed job
                config_id = os.path.basename(file_path).replace('.json', '')
                failed_result = {
                    'config_id': config_id,
                    'uq_method': 'FAILED',
                    'total_params': 0,
                    'trainable_params': 0,
                    'model_size_mb': float('nan'),
                    'uncertainty_estimating_throughput': float('nan'),
                    'throughput_std': float('nan'),
                    'max_memory_usage': float('nan'),
                    'experiment_timestamp': '',
                    'base_architecture': 'FAILED',
                    'dataset_name': 'FAILED',
                }
                combined_results.append(failed_result)
                
        except Exception as e:
            # Handle any other errors during processing
            config_id = os.path.basename(file_path).replace('.json', '')
            error_result = {
                'config_id': config_id,
                'uq_method': 'ERROR',
                'total_params': 0,
                'trainable_params': 0,
                'model_size_mb': float('nan'),
                'uncertainty_estimating_throughput': float('nan'),
                'throughput_std': float('nan'),
                'max_memory_usage': float('nan'),
                'experiment_timestamp': '',
                'base_architecture': 'ERROR',
                'dataset_name': str(e)[:100],  # Truncate error message
            }
            combined_results.append(error_result)
    
    # Convert to DataFrame and save
    if combined_results:
        combined_df = pd.DataFrame(combined_results)
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        combined_df.to_csv(output_file, index=False)
        print(f"Combined {len(combined_results)} results into {output_file}")
    else:
        print("No results to combine")
    
    return output_file


@click.command()
@click.option('--param_config', required=True, help='Path to parameter configuration JSON file from generate_parameter_grid.py')
@click.option('--config', default='config.yaml', help='Path to the base config file')
@click.option('--output', default='throughput_results', help='Path to the output directory')
@click.option('--parsl_rundir', default='./rundir', help='Path to the parsl run directory')
@click.option('--dataset_name', default='five_d_uniform', help='Dataset name from config')
@click.option('--model_name', default='resnet_50_size', help='Model architecture name from config')
@click.option('--max_tasks', default=None, type=int, help='Maximum number of tasks to run (for testing)')
@click.option('--local', is_flag=True, help='Run tasks locally instead of on the cluster')
@click.option('--skip-completed', is_flag=True, help='Skip tasks if their output file already exists')
def main(param_config, config, output, parsl_rundir, dataset_name, model_name, max_tasks, local, skip_completed):
    config_filename = config
    
    print(f"Loading parameter configurations from {param_config}")
    param_data = load_parameter_configs(param_config)
    
    print(f"Loading base configuration from {config}")
    config_data = get_config(config)
    
    print(f"Using model architecture: {model_name}")
    
    slurm_settings = config_data.get('throughput_slurm_config', {})
    
    local_provider = LocalProvider(
        init_blocks=1,
        max_blocks=1,
        parallelism=1
    )
    
    gpus_per_task = slurm_settings.pop('gpus_per_task', 1)
    cpus_per_gpu = slurm_settings.pop('cpus_per_gpu', 16)
    nodes = slurm_settings.pop('nodes', 1)
    ntasks_per_node = slurm_settings.pop('ntasks_per_node', 1)
    
    scheduler_opts = (
        f"--gpus-per-task={gpus_per_task} "
        f"--cpus-per-gpu={cpus_per_gpu} "
        f"--nodes={nodes} "
        f"--ntasks-per-node={ntasks_per_node}"
    )
    slurm_settings['scheduler_options'] = f"#SBATCH {scheduler_opts}"
    
    slurm_provider = SlurmProvider(
        **slurm_settings,
        launcher=SingleNodeLauncher()
    )
    
    provider = local_provider if local else slurm_provider
    
    # Configure Parsl
    parsl_config = Config(
        retries=3,
        run_dir=parsl_rundir,
        executors=[
            HighThroughputExecutor(
                cores_per_worker=16,
                available_accelerators=1,
                cpu_affinity='block',
                mem_per_worker=64,
                worker_debug=False,
                label="Size_vs_Throughput_Exec",
                provider=provider
            )
        ]
    )
    parsl.load(parsl_config)
    
    # Clear SLURM environment variables to avoid conflicts
    try:
        os.unsetenv('SLURM_CPU_BIND')
        os.unsetenv('SLURM_CPU_BIND_LIST')
        os.unsetenv('SLURM_CPUS_ON_NODE')
        os.unsetenv('SLURM_CPUS_PER_TASK')
        os.unsetenv('SLURM_CPU_BIND_TYPE')
        os.unsetenv('SLURM_JOB_NAME')
    except KeyError:
        pass
    
    # Get list of configurations to run
    configurations = param_data.get('configurations', [])
    total_configs = len(configurations)
    
    print(f"Found {total_configs} configurations to run")
    
    # Create output directory
    os.makedirs(output, exist_ok=True)
    os.makedirs(f"{output}/individual_results", exist_ok=True)
    
    # Filter out completed tasks and prepare result files list
    uncompleted_configs = []
    result_files = []
    
    for config in configurations:
        config_id = config.get('config_id')
        output_file = f"{output}/individual_results/{config_id}.json"
        result_files.append(output_file)
        
        if skip_completed and os.path.exists(output_file):
            print(f"Skipping completed task: {config_id} (output file exists: {output_file})")
            continue
        else:
            uncompleted_configs.append(config)
    
    print(f"Found {len(uncompleted_configs)} uncompleted configurations")
    
    # Limit the number of new tasks to submit if max_tasks is specified
    configs_to_submit = uncompleted_configs
    if max_tasks is not None:
        configs_to_submit = uncompleted_configs[:max_tasks]
        print(f"Limiting to {max_tasks} new tasks to submit (out of {len(uncompleted_configs)} uncompleted)")
    
    # Submit throughput experiment tasks
    experiment_results = []
    
    for config in configs_to_submit:
        config_id = config.get('config_id')
        output_file = f"{output}/individual_results/{config_id}.json"
        
        print(f'Submitting throughput experiment for config ID: {config_id} with model: {model_name}')
        res = run_throughput_experiment(
            config_file=config_filename,
            param_config_file=param_config,
            config_id=config_id,
            dataset_name=dataset_name,
            model_name=model_name,
            output_file=output_file
        )
        experiment_results.append(res)
    
    # Wait for all submitted tasks to complete
    print(f"Waiting for {len(experiment_results)} submitted tasks to complete...")
    for i, res in enumerate(experiment_results):
        try:
            res.result()
            print(f"Task {i+1}/{len(experiment_results)} completed")
        except Exception as e:
            print(f"Task {i+1}/{len(experiment_results)} failed with error: {e}")
    
    # Combine all results into a single CSV file
    print("Combining all results...")
    combined_output = f"{output}/all_throughput_results.csv"
    combine_task = combine_throughput_results(result_files, combined_output)
    final_output = combine_task.result()
    
    print(f"\n{'='*60}")
    print("THROUGHPUT WORKFLOW COMPLETED")
    print(f"{'='*60}")
    print(f"Total configurations found: {total_configs}")
    print(f"Uncompleted configurations: {len(uncompleted_configs)}")
    print(f"New tasks submitted: {len(experiment_results)}")
    print(f"Individual results saved in: {output}/individual_results/")
    print(f"Combined results saved to: {final_output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
