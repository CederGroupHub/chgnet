## Model r2SCAN

This is the pretrained weights of CHGNet fine-tuned on the MP-r2SCAN dataset. The model was initialized from the GGA/GGA+U trained CHGNet v0.3.0 and then transferred to the R2SCAN functional dataset. This work is published in the npj Computational Materials paper titled "Cross-functional transferability in foundation machine learning interatomic potentials".

All experiments and results shown in the paper (Method 4) were performed with this version of weights.

Date: 9/21/2025

Author: Xu Huang

If you use CHGNet_r2scan or MP-r2SCAN dataset, please cite <a href="https://www.nature.com/articles/s41524-025-01796-y" target="_blank">this paper</a>.

## Model Parameters

```python
model = CHGNet(
    atom_fea_dim=64,
    bond_fea_dim=64,
    angle_fea_dim=64,
    composition_model="MP-r2SCAN",
    num_radial=31,
    num_angular=31,
    n_conv=4,
    atom_conv_hidden_dim=64,
    update_bond=True,
    bond_conv_hidden_dim=64,
    update_angle=True,
    angle_layer_hidden_dim=0,
    conv_dropout=0,
    read_out="ave",
    gMLP_norm='layer',
    readout_norm='layer',
    mlp_hidden_dims=[64, 64, 64],
    mlp_first=True,
    is_intensive=True,
    non_linearity="silu",
    atom_graph_cutoff=6,
    bond_graph_cutoff=3,
    graph_converter_algorithm="fast",
    cutoff_coeff=8,
    learnable_rbf=True,
)
```

## Dataset Used

MP-r2SCAN dataset (https://doi.org/10.6084/m9.figshare.28245650.v2) with 8-1-1 train-val-test splitting

## Load the Model

```python
chgnet = CHGNet.load(model_name='r2scan')
```

## Training Configuration

We used the pretrained CHGNet v0.3.0 as the starting model and fine-tuned it on the MP-r2SCAN dataset.

```python
# Load pretrained CHGNet v0.3.0
chgnet = CHGNet.load()

# Update model_args to reflect the new composition model
chgnet.model_args['composition_model'] = "MP-r2SCAN"

# Reinitialize composition model weights for MP-r2SCAN dataset
chgnet.composition_model.initialize_from("MP-r2SCAN")

# Initialize trainer with specific configuration
trainer = Trainer(
    model=chgnet,
    targets='efsm',
    energy_loss_ratio=3,
    force_loss_ratio=1,
    stress_loss_ratio=0.1,
    mag_loss_ratio=1,
    optimizer='Adam',
    scheduler='CosLR',
    criterion='Huber',
    epochs=50,
    learning_rate=1e-3,
    use_device='cuda'
)

# Fine-tune the model
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    save_dir=save_dir,
    train_composition_model=False
)
```

## Mean Absolute Error (MAE) logs

| partition  | Energy (meV/atom) | Force (meV/A) | stress (GPa) | magmom (muB) |
| ---------- | ----------------- | ------------- | ------------ | ------------ |
| Train      | 11.82             | 24.55         | 0.082        | 0.021        |
| Validation | 15.48             | 36.50         | 0.161        | 0.023        |
| Test       | 16.76             | 38.46         | 0.167        | 0.023        |
