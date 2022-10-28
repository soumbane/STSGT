# Spatial–Temporal Synchronous Graph Transformer network (STSGT) for COVID-19 forecasting

This is the official implementation of the paper "[Spatial–Temporal Synchronous Graph Transformer network (STSGT) for COVID-19 forecasting](https://www.sciencedirect.com/science/article/pii/S2352648322000824)" that is published in Smart Health Journal

## Requirements
* Python >= 3.6
* PyTorch >= 1.8.0
* See requirements.txt (All of them are not required)

## Get Started
1. Convert multiple datasets to a `magnet.data.TargetDataset` and use `magnet.data.TargetedDataLoader` to load the data
```
import ...
import magnet

# load data
train_dataset_1, val_dataset_1 = ...
train_dataset_2, val_dataset_2 = ...
...
target_dict = {0: "m1", 1: "m2", ...}
training_dataset = magnet.data.TargetedDataset(train_dataset_1, train_dataset_2, target_dict=target_dict)
training_dataset = magnet.data.TargetedDataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
validation_dataset = {
    "m1": data.DataLoader(val_dataset_1, batch_size=batch_size, shuffle=False),
    "m2": data.DataLoader(val_dataset_2, batch_size=batch_size, shuffle=False),
	...
}
```

