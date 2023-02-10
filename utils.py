import torch
import torch.nn as nn
from modules import *


def isActivation(name):
    if 'relu' in name.lower() or 'qcfs' in name.lower():
        return True
    return False
    
    
def replace_MPLayer_by_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_MPLayer_by_neuron(module)
        if module.__class__.__name__ == 'MPLayer':
            model._modules[name] = IFNeuron(scale=module.v_threshold)
    return model


def replace_activation_by_MPLayer(model, presim_len, sim_len, batchsize):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_MPLayer(module, presim_len, sim_len, batchsize)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = MPLayer(v_threshold=module.up.item(), presim_len=presim_len, sim_len=sim_len, batchsize=batchsize)
    return model


def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model



def replace_activation_by_floor(model, t):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_floor(module, t)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = QCFS(up=8., t=t)
    return model


def reset_net(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            reset_net(module)
        if 'Neuron' in module.__class__.__name__:
            module.reset()
    return model


def calculate_MPLayer(model,ans):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            calculate_MPLayer(module,ans)
        if module.__class__.__name__ == 'MPLayer':
            tot_input, tot_one, tot_two, tot_three, tot_zero = 0,0,0,0,0
            tot_len = len(module.arr2) / 4.
            for i in range(0,len(module.arr2),4):
                tot_one += module.arr2[i+1]
                tot_two += module.arr2[i+2]
                tot_three += module.arr2[i+3]
                tot_zero += module.arr2[i]
                     
            ans.append(tot_zero.item()/tot_len)
            ans.append(tot_one.item()/tot_len)
            ans.append(tot_two.item()/tot_len)
            ans.append(tot_three.item()/tot_len)
            module.arr = []
            module.arr2 = []

    return model

def set_MPLayer(model,snn):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            set_MPLayer(module,snn)
        if module.__class__.__name__ == 'MPLayer':
            module.snn_mode = snn
    return model


def error(info):
    print(info)
    exit(1)
    