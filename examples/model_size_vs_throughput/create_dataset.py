#!/usr/bin/env python3

import click
import numpy as np
import os

@click.command()
@click.option('--size', '-s', default=10000, help='Number of data points to generate')
@click.option('--output', '-o', default='dataset.npy', help='Output filename for the numpy dataset')
@click.option('--seed', default=None, type=int, help='Random seed for reproducibility')
def create_dataset(size, output, seed):
    """
    Create a numpy dataset with 4 input dimensions and 1 output dimension.
    All data is uniformly random in [0, 1] with double precision.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate uniformly random data in [0, 1] with double precision
    # Shape: (size, 5) where first 4 columns are inputs, last column is output
    data = np.random.uniform(0.0, 1.0, size=(size, 5)).astype(np.float64)
    
    # Save the dataset
    np.save(output, data)
    
    click.echo(f"Dataset created successfully!")
    click.echo(f"  Size: {size} samples")
    click.echo(f"  Shape: {data.shape}")
    click.echo(f"  Dtype: {data.dtype}")
    click.echo(f"  Input dimensions: 4")
    click.echo(f"  Output dimensions: 1")
    click.echo(f"  Data range: [0, 1]")
    click.echo(f"  Saved to: {output}")
    
    # Print some basic statistics
    click.echo(f"\nData statistics:")
    click.echo(f"  Min value: {data.min():.6f}")
    click.echo(f"  Max value: {data.max():.6f}")
    click.echo(f"  Mean value: {data.mean():.6f}")
    click.echo(f"  File size: {os.path.getsize(output) / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    create_dataset() 