#!/usr/bin/env python
# coding: utf-8
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["PYTHONWARNINGS"] = "ignore"  # 全局忽略所有 warnings


import numpy as np
import torch
import torch.nn as nn
import logging
import datetime
from pathlib import Path

import cace
from cace.representations import Cace
from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered

from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask

torch.set_default_dtype(torch.float32)

cace.tools.setup_logger(level='INFO')
cutoff = 5.0

#val_ratio = float(sys.argv[1])
MP = 1
N_dl = 2
save_folder = "/data/home/public/qiuqizhi/SOG-Qeq/SOG-Net/CACE-SOG-Ji/fit-cumulene-out3-change/loss_data/SOG_MP"+str(MP)+"_"
now = datetime.datetime.now()
time_name=now.strftime("%Y%m%d_%H%M%S")
os.makedirs(os.path.dirname(save_folder), exist_ok=True)

print("reading data")
dataset_root = Path("/data/home/public/qiuqizhi/LOREM/datasets")
collection = cace.tasks.get_dataset_from_xyz(
    train_path=str(dataset_root / "cumulene_train.xyz"),
    valid_path=str(dataset_root / "cumulene_valid.xyz"),
    test_path=str(dataset_root / "cumulene_test.xyz"),
    seed=42,
    cutoff=cutoff,
    data_key={'energy': 'energy', 'forces': 'forces'},
    # CACE cumulene baseline (same as LOREM cace script)
    atomic_energies={1: -2.6544, 6: -5.9724},
)
batch_size = 8

train_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='train',
                              batch_size=batch_size,
                              )

valid_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='valid',
                              batch_size=8,
                              )

test_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='test',
                              batch_size=8,
                              )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = cace.tools.init_device(use_device)
print(f"device: {device}")


print("building CACE representation")
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
#cutoff_fn = CosineCutoff(cutoff=cutoff)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

cace_representation = Cace(
    zs=[1, 6],
    n_atom_basis=8,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=3,
    max_nu=3,
    num_message_passing=MP,
    type_message_passing=['Bchi'],
    args_message_passing={'Bchi': {'shared_channels': True, 'shared_l': True}},
    #avg_num_neighbors=1,
    device=device,
    timeit=False
           )

cace_representation.to(device)
print(f"Representation: {cace_representation}")

atomwise = cace.modules.atomwise.Atomwise(n_layers=3,
                                         output_key='CACE_energy',
                                         n_hidden=[1,1],
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
    n_hidden=[1,1],
    n_out=3,
    per_atom_output_key='q',
    output_key = 'tot_q',
    residual=False,
    add_linear_nn=True,
    bias=False)

ep = cace.modules.SOGPotential(N_dl =N_dl,
                    feature_key='q',
                    output_key='SOG_potential',
                    remove_self_interaction=False,
                   aggregation_mode='sum',
                   Periodic = False)

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
    loss_weight=1000.0
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

boost = 1

optimizer_args = {'lr': 5e-3, 'betas': (0.99, 0.999)}
scheduler_args = {'step_size': 20, 'gamma': 0.5}

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

# Total epochs = 500 with staged energy loss weight:
# 0-200: 0.1, 200-300: 1, 300-400: 10, 400-500: 1000
stage_plan = [
    (200, 0.1),
    (100, 1.0),
    (100, 10.0),
    (100, 1000.0),
]
for stage_idx, (epochs_now, energy_weight_now) in enumerate(stage_plan, start=1):
    print(f"Stage {stage_idx}: epochs={epochs_now}, energy_weight={energy_weight_now}, force_weight=1000.0")
    stage_energy_loss = cace.tasks.GetLoss(
        target_name='energy',
        predict_name='CACE_energy',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=energy_weight_now
    )
    task.update_loss([stage_energy_loss, force_loss])
    task.fit(train_loader, valid_loader, epochs=epochs_now, screen_nan=False, val_stride=10)

task.save_model(save_folder+'cumulene-model-final.pth')

print("Evaluating on cumulene test set")
test_loss = task.validate(test_loader)
print(f"test_loss: {test_loss:.6f}")

# print(f"Fourth train loop:")
# energy_loss = cace.tasks.GetLoss(
#     target_name='energy',
#     predict_name='CACE_energy',
#     loss_fn=torch.nn.MSELoss(),
#     loss_weight=0.001
# )

# task.update_loss([energy_loss, force_loss])
# task.fit(train_loader, valid_loader, epochs=50*boost, screen_nan=False, val_stride=10)

# task.save_model('electrolyte-model-4.pth')

print(f"Finished")


trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")



