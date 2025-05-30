import click
import os
import parsl
import yaml
from itertools import product
from parsl.app.app import python_app
from parsl.app.app import join_app
from parsl.app.app import bash_app
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.providers import LocalProvider
from parsl.launchers import SrunLauncher
from parsl.data_provider.files import File as ParslFile
from parsl import set_stream_logger
from parsl.launchers import SingleNodeLauncher
from parsl.executors import HighThroughputExecutor

def get_config(config_filename):
    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)
    return config


@bash_app(cache=True)
def run_bo(config, benchmark, uq_method, dataset, output,
           stdout=parsl.AUTO_LOGNAME,
           stderr=parsl.AUTO_LOGNAME):
    import sh
    import os
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
    command = python.bake('bo.py', '--benchmark', benchmark, 
                          '--config', config,
                          '--uq_method', uq_method, 
                          '--dataset', dataset, 
                          '--output', output,
                          '--restart'
                          )
    print(str(command))
    return str(command)

@click.command()
@click.option('--config', default='./config.yaml', help='Path to the config file', required=False)
@click.option('--output', default='workflow_output', help='Path to the output directory.', required=False)
@click.option('--parsl_rundir', default='./rundir', help='Path to the parsl run directory', required=False)
def main(config, output, parsl_rundir):

    config_filename = config
    # Load config first to get Slurm settings
    config_data = get_config(config_filename)
    slurm_settings = config_data.get('bo_slurm_config', {}) # Get BO slurm config

    local_provider = LocalProvider(
        init_blocks=1,
        max_blocks=1,
        parallelism=1
    )

    # Construct scheduler options string dynamically and add to settings
    gpus_per_task = slurm_settings.pop('gpus_per_task', 1) # Remove key after getting value
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
        # Unpack settings from the config file
        **slurm_settings,
        # Ensure launcher is always set, potentially overriding config if 'launcher' key exists
        launcher=SingleNodeLauncher()
    )

    parsl_config = Config(
        retries=20,
        run_dir=parsl_rundir,
        executors=[
            HighThroughputExecutor(
                cores_per_worker=16, # Revert to hardcoded value
                available_accelerators=1, # Revert to hardcoded value
                cpu_affinity='block',
                mem_per_worker=64, # Revert to hardcoded value
                worker_debug=False,
                label="BO_Search_Exec",
                provider=slurm_provider
            )
        ]
    )
    parsl.load(parsl_config)

    # config = get_config(config_filename) # Config is already loaded as config_data
    benches = config_data['benchmarks'].keys()
    uq_methods = config_data['uq_methods'].keys()
    dsets = ['tails', 'gaps']

    total = list(product(benches, uq_methods, dsets))

    # This causes issues when submitting one job from another, see:
    # https://bugs.schedmd.com/show_bug.cgi?id=14298
    try:
        os.unsetenv('SLURM_CPU_BIND')
        os.unsetenv('SLURM_CPU_BIND_LIST')
        os.unsetenv('SLURM_CPUS_ON_NODE')
        os.unsetenv('SLURM_CPUS_PER_TASK')
        os.unsetenv('SLURM_CPU_BIND_TYPE')
        os.unsetenv('SLURM_JOB_NAME')
    except KeyError:
        pass

    results = list()
    for bench, uq_method, dset in total:
        print(f'Running {bench} with {uq_method} on {dset}')
        res = run_bo(config_filename, bench, uq_method, dset, output)
        results.append(res)

    for res in results:
        print(res.result())


if __name__ == '__main__':
    main()
