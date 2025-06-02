# Learning Rate Scheduler Implementation

This document describes the new learning rate scheduler functionality that replaces the manual epoch-based learning rate specification.

## Overview

The learning rate scheduler functionality has been implemented to provide more flexible and automated learning rate scheduling during training. This replaces the previous manual epoch-based learning rate dictionary approach.

## Changes Made

### 1. Configuration Changes (`neuralhydrology/utils/config.py`)

- **Modified `learning_rate` property**: Now returns a single `float` value representing the initial learning rate, instead of a dictionary mapping epochs to learning rates.
- **Added `lr_scheduler` property**: New configuration option for specifying learning rate scheduler parameters.

### 2. Training Infrastructure (`neuralhydrology/training/__init__.py`)

- **Updated `get_optimizer` function**: Modified to use the new single-value learning rate.
- **Added `get_lr_scheduler` function**: Creates and configures learning rate schedulers based on configuration.

### 3. Training Loop (`neuralhydrology/training/basetrainer.py`)

- **Updated `BaseTrainer` class**: Added `lr_scheduler` attribute and initialization.
- **Modified `train_and_validate` method**: Removed manual learning rate setting and added automatic scheduler stepping.
- **Added learning rate logging**: Current learning rate is logged to tensorboard/wandb automatically.

### 4. Logging Support (`neuralhydrology/training/wandb_logger.py`)

- **Added `log_lr` method**: Logs current learning rate values to wandb.

## Usage

### Configuration Format

Replace the old epoch-based learning rate specification:

```yaml
# Old format (deprecated)
learning_rate:
  0: 0.001
  10: 0.0005
  20: 0.0001
```

With the new format:

```yaml
# New format
learning_rate: 0.001  # Initial learning rate

# Optional: Learning rate scheduler
lr_scheduler:
  type: StepLR
  step_size: 10
  gamma: 0.5
```

### Supported Schedulers

The following PyTorch learning rate schedulers are supported:

1. **StepLR**
   ```yaml
   lr_scheduler:
     type: StepLR
     step_size: 10      # Required
     gamma: 0.1         # Optional (default: 0.1)
   ```

2. **MultiStepLR**
   ```yaml
   lr_scheduler:
     type: MultiStepLR
     milestones: [10, 20, 30]  # Required
     gamma: 0.1                # Optional (default: 0.1)
   ```

3. **ExponentialLR**
   ```yaml
   lr_scheduler:
     type: ExponentialLR
     gamma: 0.95  # Required
   ```

4. **CosineAnnealingLR**
   ```yaml
   lr_scheduler:
     type: CosineAnnealingLR
     T_max: 50          # Required
     eta_min: 0         # Optional (default: 0)
   ```

5. **ReduceLROnPlateau**
   ```yaml
   lr_scheduler:
     type: ReduceLROnPlateau
     mode: min          # Optional (default: 'min')
     factor: 0.1        # Optional (default: 0.1)
     patience: 10       # Optional (default: 10)
     threshold: 0.0001  # Optional (default: 1e-4)
   ```

6. **CosineAnnealingWarmRestarts**
   ```yaml
   lr_scheduler:
     type: CosineAnnealingWarmRestarts
     T_0: 10           # Required
     T_mult: 1         # Optional (default: 1)
     eta_min: 0        # Optional (default: 0)
   ```

7. **LinearLR**
   ```yaml
   lr_scheduler:
     type: LinearLR
     start_factor: 1.0   # Optional (default: 1.0)
     total_iters: 100    # Required
   ```

8. **PolynomialLR**
   ```yaml
   lr_scheduler:
     type: PolynomialLR
     total_iters: 100  # Required
     power: 1.0        # Optional (default: 1.0)
   ```

### No Scheduler

If no learning rate scheduler is desired, simply omit the `lr_scheduler` configuration:

```yaml
learning_rate: 0.001
# No lr_scheduler specified - constant learning rate will be used
```

## Backward Compatibility

For backward compatibility, if a dictionary is provided for `learning_rate`, the first value will be used as the initial learning rate:

```yaml
# This will work but is deprecated
learning_rate:
  0: 0.001
  10: 0.0005
# Will use 0.001 as initial learning rate, ignoring epoch-based changes
```

## Benefits

1. **Automated scheduling**: No need to manually specify learning rates for different epochs.
2. **Rich scheduler options**: Support for all common PyTorch learning rate schedulers.
3. **Automatic logging**: Learning rate changes are automatically logged to tensorboard/wandb.
4. **Flexible configuration**: Easy to experiment with different scheduling strategies.
5. **Validation-aware scheduling**: ReduceLROnPlateau can use validation loss for scheduling decisions.

## Migration Guide

To migrate existing configurations:

1. Replace the learning rate dictionary with a single initial learning rate value.
2. Add an appropriate `lr_scheduler` configuration based on your previous schedule.
3. Test the new configuration to ensure similar behavior.

Example migration:

```yaml
# Before
learning_rate:
  0: 0.001
  10: 0.0005
  20: 0.0001

# After
learning_rate: 0.001
lr_scheduler:
  type: MultiStepLR
  milestones: [10, 20]
  gamma: 0.5  # 0.001 * 0.5 = 0.0005, 0.0005 * 0.5 = 0.00025 ≈ 0.0001
``` 