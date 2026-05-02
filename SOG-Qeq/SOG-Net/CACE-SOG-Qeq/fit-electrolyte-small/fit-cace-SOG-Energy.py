#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../cace/')

import numpy as np
import torch
import torch.nn as nn
import logging
import datetime

import cace
from cace.representations import Cace
from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered

from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask

torch.set_default_dtype(torch.float32)

cace.tools.setup_logger(level='INFO')
cutoff = 4.5
MP = 1
N_dl = 1
save_folder = "/dssg/home/acct-matxzl/matxzl/Yajie/MDNN/cace-lr-fit-main_dl/fit-electrolyte-B/loss_data/SGcutoff_"+str(cutoff)+"_Energy"
now = datetime.datetime.now()
time_name=now.strftime("%Y%m%d_%H%M%S")

#val_ratio = float(sys.argv[1])
val_ratio = 0.1

print("reading data")
collection = cace.tasks.get_dataset_from_xyz(train_path='/dssg/home/acct-matxzl/matxzl/Yajie/MDNN/cace-lr-fit-main/fit-electrolyte-B/electrolyte.xyz',
                                 valid_fraction=val_ratio,
                                 seed=1,
                                 cutoff=cutoff,
                                 data_key={'energy': 'energy', 'forces':'forces'}, 
                                 atomic_energies={1: -0.1749365806299343, 8: -0.08746829031496617, 9: -4.620975436064299, 19: -4.620975436064285} # avg
                                 )
batch_size = 5

train_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='train',
                              batch_size=batch_size,
                              )

valid_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='valid',
                              batch_size=10,
                              )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = cace.tools.init_device(use_device)
print(f"device: {device}")


print("building CACE representation")
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
#cutoff_fn = CosineCutoff(cutoff=cutoff)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

cace_representation = Cace(
    zs=[1, 8, 9, 19],
    n_atom_basis=4,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=3,
    max_nu=3,
    num_message_passing=MP,
    type_message_passing=['Bchi'],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    #avg_num_neighbors=1,
    device=device,
    timeit=False
           )

cace_representation.to(device)
print(f"Representation: {cace_representation}")

atomwise = cace.modules.atomwise.Atomwise(n_layers=3,
                                         output_key='CACE_energy',
                                         n_hidden=[32,16],
                                         use_batchnorm=False,
                                         add_linear_nn=True)

forces = cace.modules.forces.Forces(energy_key='CACE_energy',
                                    forces_key='CACE_forces')

print("building CACE NNP")
cace_nnp_sr = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[atomwise, forces]
)

q = cace.modules.Atomwise(
    n_layers=3,
    n_hidden=[24,12],
    n_out=1,
    per_atom_output_key='q',
    output_key = 'tot_q',
    residual=False,
    add_linear_nn=True,
    bias=False)

ep = cace.modules.SOGPotential(N_dl=N_dl,
                    feature_key='q',
                    output_key='SOG_potential',
                    remove_self_interaction=False,
                    aggregation_mode='sum',
                    Periodic=True)

forces_lr = cace.modules.Forces(energy_key='SOG_potential',
                                    forces_key='SOG_forces')

cace_nnp_lr = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[q, ep, forces_lr]
)

pot2 = {'CACE_energy': 'SOG_potential', 
        'CACE_forces': 'SOG_forces',
        'weight': 1
       }

pot1 = {'CACE_energy': 'CACE_energy', 
        'CACE_forces': 'CACE_forces',
       }

cace_nnp = cace.models.CombinePotential([cace_nnp_sr, cace_nnp_lr], [pot1,pot2])
#cace_nnp = cace.models.CombinePotential([cace_nnp_sr], [pot1])
cace_nnp.to(device)


print(f"First train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.1
)

force_loss = cace.tasks.GetLoss(
    target_name='forces',
    predict_name='CACE_forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

from cace.tools import Metrics

e_metric = Metrics(
    target_name='energy',
    predict_name='CACE_energy',
    name='e/atom',
    per_atom=True
)

f_metric = Metrics(
    target_name='forces',
    predict_name='CACE_forces',
    name='f'
)

# Example usage
print("creating training task")

boost = int((1./(1.-val_ratio))**0.5)

optimizer_args = {'lr': 1e-2, 'betas': (0.99, 0.999)}  
scheduler_args = {'step_size': 20*boost, 'gamma': 0.8}

for i in range(5):
    task = TrainingTask(
        model=cace_nnp,
        losses=[energy_loss, force_loss],
        metrics=[e_metric, f_metric],
        device=device,
        optimizer_args=optimizer_args,
        scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=False, #True,
        ema_start=10,
        warmup_steps=5,
        save_folder = save_folder,
        time_name = time_name
    )

    print("training")
    task.fit(train_loader, valid_loader, epochs=40*boost, screen_nan=False, val_stride=10)

task.save_model('electrolyte-model-SOG-Ener.pth')
cace_nnp.to(device)

print(f"Second train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1
)

task.update_loss([energy_loss, force_loss])
print("training")
task.fit(train_loader, valid_loader, epochs=50*boost, screen_nan=False, val_stride=10)

task.save_model('electrolyte-model-SOG-2-Ener.pth')
cace_nnp.to(device)

print(f"Third train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=10 
)

task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=50*boost, screen_nan=False, val_stride=10)

task.save_model('electrolyte-model-SOG-3-Ener.pth')

print(f"Fourth train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=50*boost, screen_nan=False, val_stride=10)

task.save_model('electrolyte-model-SOG-4-Ener.pth')

print(f"Finished")


trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")



