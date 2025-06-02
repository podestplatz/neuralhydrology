#!/usr/bin/env python3
"""
This script configures a sweep and creates it on WandB. 

Use nh_run.py in sweep mode to add an agent to the sweep. An agent will then execute as long as there are configurations
left to try in the sweep. 
"""
import wandb

# This specifies the hyperparameters to sweep over
sweep_config = {
    'method': 'bayes',  # Can be 'grid', 'random', or 'bayes'
    'name': 'optimise_satimg_approach',  # Give your sweep a meaningful name
    'metric': {
        'name': 'valid/avg_total_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'value': 0.001
        },
        'lr_scheduler': {
            'values': [
                {
                    'type': 'CosineAnnealingLR',
                    'T_max': 50,
                    'eta_min': 1e-5
                },
                {
                    'type': 'ReduceLROnPlateau',
                    'factor': 0.5,
                    'patience': 5,
                    'min_lr': 1e-5
                }, 
                {
                    'type': 'ExponentialLR',
                    'gamma': 0.95
                },
            ]
        },
        'hidden_size': {
            'values': [64, 128, 256]  # actually sensible
        },
        'output_dropout': {
            'values': [0.0, 0.1, 0.2]  # sweep up to 0.5
        },
        'epochs': {  # maybe remove it, or change in to a higher value
            'value': 50  # Fixed value
        }, 
        'target_noise_std': {
            'values': [0.0, 0.001, 0.005, 0.01, 0.05]
        },
        'batch_size': {
            'values': [64, 128, 256]
        }
    }
}

def create_sweep():
    """Create a wandb sweep and return the sweep ID.
    
    Returns
    -------
    str
        The sweep ID that can be used to run agents.
    """
    # Initialize wandb project
    wandb.login()
    
    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="neuralhydrology")
    
    print(f"Created sweep with ID: {sweep_id}")
    print(f"To run sweep agents, use:")
    print(f"wandb agent {sweep_id}")
    print(f"OR use the neuralhydrology sweep mode:")
    print(f"python ../../neuralhydrology/nh_run.py sweep --config-file /path/to/your/base_config.yml")
    print()
    print("Sweep configurations will be saved in a 'sweep_<sweep_name>' directory next to your base config file.")
    
    return sweep_id

if __name__ == "__main__":
    sweep_id = create_sweep() 