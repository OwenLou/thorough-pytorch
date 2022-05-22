# Task 03

Summarize useful techiques.

## Model definition

```Python
nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
    )

nn.Sequential(collections.OrderedDict([
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 10))
    ]))
```

- below ones need `forward` method to be defined.
```Python
nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])

nn.ModuleDict({'linear': nn.Linear(784, 256), 'act': nn.ReLU(),})
```

- Better name each layer with a unique name.



## Fast modeling

Use derived class of `nn.Module` to define consiced layers.



## Model modification

- Model copy: use `import copy; net_cp = copy.deepcopy(net)`
- Then change the class instance elements.



## Model saving and loading

- Save and load `state_dict` suggested.
  - Always work for either cpu or gpu (either multi or single).
- If model saved instead of state_dict, then extract the state_dict from the model and then load the state_dict.
  - To avoid the problem of gpu confliction.

```Python
"""single gpu or cpu"""
torch.save(unet.state_dict(), "./unet_weight_example.pth")
loaded_unet_weights = torch.load("./unet_weight_example.pth")
unet.load_state_dict(loaded_unet_weights)
unet.state_dict()
```

- Multi-card training
  - The model is transfered to `module`

```Python
"""multi gpu"""
torch.save(unet_mul.state_dict(), "./unet_weight_mul_example.pth")
loaded_unet_weights_mul = torch.load("./unet_weight_mul_example.pth")
unet_mul.load_state_dict(loaded_unet_weights_mul)
unet_mul = nn.DataParallel(unet_mul).cuda()
unet_mul.state_dict()
```

## Loss function

- Just use derived class `nn.Module`



## Dynamic learning rate

- `optim.lr_scheduler.StepLR`
  - `step_size`: how many epochs to decay.
  - `gamma`: Multiplicative factor of learning rate decay. Default: 0.1.
  - `last_epoch`: The index of last epoch. Default: -1.

```Python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
```



## Fine tuning

- Use `requires_grad`
  - `nn.Parameter.requires_grad`: `True` or `False`

```Python
unet.module.outc.conv.weight.requires_grad = False
unet.module.outc.conv.bias.requires_grad = False

for layer, param in unet.named_parameters():
    print(layer, '\t', param.requires_grad)
```


## Float16 training


