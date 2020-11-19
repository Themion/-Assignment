# -*- coding: utf-8 -*-
"""
Created on Sun May 31 20:38:03 2020

@author: RML
"""

import torch

import torchvision.models as models

def _cifarnet(pretrained = False, path = None):
    model = models.resnext101_32x8d()
    if pretrained:
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
    return model
