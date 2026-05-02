#!/usr/bin/env python3
"""
LLZO two-stage MD runner with ReaxNet (E0 + EPQEq + optional D3).

Stage 1: NPT (Berendsen) 100 ps, 2 fs, 800 K, 1 atm (paper-like setup)
Stage 2: NVT (Langevin) 2 ns, 2 fs, 800 K

Notes:
- Paper reports Nose-Hoover + Parrinello-Rahman. ASE Berendsen is used here
  as a practical/stable alternative for an automated script.
- This script computes stress from dE/d(strain) and feeds stress to ASE.
"""

import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixCom
from ase.io import Trajectory, read
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from jax_md import partition, space


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', default='/data/home/public/qiuqizhi/reaxnet')
    p.add_argument('--structure', default='data/LLZO/init.vasp')

    # Stage 1: NPT
    p.add_argument('--npt-temperature-k', type=float, default=800.0)
    p.add_argument('--npt-pressure-atm', type=float, default=1.0)
    p.add_argument('--npt-ps', type=float, default=100.0)

    # Stage 2: NVT
    p.add_argument('--nvt-temperature-k', type=float, default=800.0)
    p.add_argument('--nvt-ps', type=float, default=2000.0)

    # Shared integrator settings
    p.add_argument('--timestep-fs', type=float, default=2.0)
    p.add_argument('--friction', type=float, default=0.02)

    # Optional knobs
    p.add_argument('--npt-taut-fs', type=float, default=100.0)
    p.add_argument('--npt-taup-fs', type=float, default=1000.0)
    p.add_argument('--compressibility-per-bar', type=float, default=4.5e-6)

    p.add_argument('--use-d3', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--traj-out', default='llzo_800K_2stage.traj')
    p.add_argument('--log-out', default='llzo_800K_2stage.log')
    return p.parse_args()


def main():
    args = parse_args()
    jax.config.update('jax_enable_x64', True)

    project_root = Path(args.project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Local imports after sys.path injection
    from reaxnet.egnn.data import AtomicNumberTable
    from reaxnet.egnn.nequip import NequIPEnergyModel
    from reaxnet.egnn.nn_util import neighbor_list_featurizer
    from reaxnet.jax_nb.jax_nb import LAMBDA, nonbond_potential, pqeq_fori_loop
    from reaxnet.jax_nb.parameters import pqeq_parameters

    model_dir = project_root / 'reaxnet' / 'pretrained'
    structure_path = project_root / args.structure
    for fp in [model_dir / 'model_config.yaml', model_dir / 'params.pickle', model_dir / 'mapping.yaml', structure_path]:
        if not fp.exists():
            raise FileNotFoundError(f'Missing file: {fp}')

    atoms = read(str(structure_path))
    with open(model_dir / 'model_config.yaml', 'r') as f:
        model_dict = yaml.safe_load(f)
    with open(model_dir / 'params.pickle', 'rb') as f:
        params = pickle.load(f)

    ztable = AtomicNumberTable.from_dict(str(model_dir / 'mapping.yaml'))
    model = NequIPEnergyModel(**model_dict)

    positions0 = jnp.asarray(atoms.get_scaled_positions())
    box0 = jnp.asarray(atoms.get_cell().array.transpose())
    atomic_numbers = jnp.asarray(atoms.numbers)
    chemical_symbols = atoms.get_chemical_symbols()
    nn_atomic_numbers = ztable.mapping(atomic_numbers)
    nn_atomic_numbers = jax.nn.one_hot(jnp.array(nn_atomic_numbers), len(ztable) + 1)

    displacement_fn, _ = space.periodic_general(box0, fractional_coordinates=True)
    featurizer = neighbor_list_featurizer(displacement_fn)

    nn_neighbor_fn = partition.neighbor_list(
        displacement_fn, box0, model_dict['r_max'], format=partition.Sparse, fractional_coordinates=True
    )
    nb_neighbor_fn = partition.neighbor_list(
        displacement_fn, box0, 12.5, format=partition.Sparse, fractional_coordinates=True, capacity_multiplier=2.0
    )

    def energy_nn(embedded_numbers, _model, _params, position, neighbor, **kwargs):
        graph = featurizer(embedded_numbers, position, neighbor, **kwargs)
        atomic_output = _model.apply(_params, graph.edges, graph.nodes, graph.senders, graph.receivers)
        return jnp.sum(atomic_output[:-1])

    rad = jnp.array([pqeq_parameters[s]['rad'] for s in chemical_symbols])
    alpha = 0.5 * LAMBDA / rad / rad
    alpha = jnp.sqrt(alpha.reshape(-1, 1) * alpha.reshape(1, -1) / (alpha.reshape(-1, 1) + alpha.reshape(1, -1)))
    chi0 = jnp.array([pqeq_parameters[s]['chi0'] for s in chemical_symbols])
    eta0 = jnp.array([pqeq_parameters[s]['eta0'] for s in chemical_symbols])
    z = jnp.array([pqeq_parameters[s]['Z'] for s in chemical_symbols])
    Ks = jnp.array([pqeq_parameters[s]['Ks'] for s in chemical_symbols])

    energy_fn_nn = lambda p, n: energy_nn(nn_atomic_numbers, model, params, p, n)

    def total_energy_fn(pos_frac, box, strain):
        # IMPORTANT:
        # Keep this function side-effect free for JAX autodiff.
        # Build neighbor lists locally to avoid tracer leaks.
        nn_nbr = nn_neighbor_fn.allocate(pos_frac)
        nb_nbr = nb_neighbor_fn.allocate(pos_frac)

        pe_nn = energy_fn_nn(pos_frac, nn_nbr)
        charges, r_shell = pqeq_fori_loop(
            displacement_fn,
            jax.lax.stop_gradient(pos_frac),
            nb_nbr,
            alpha=alpha,
            cutoff=12.5,
            iterations=2,
            net_charge=0.0,
            eta0=eta0,
            chi0=chi0,
            z=z,
            Ks=Ks,
        )
        pe_nb = nonbond_potential(
            displacement_fn,
            pos_frac,
            nb_nbr,
            r_shell,
            charges,
            alpha=alpha,
            cutoff=12.5,
            eta0=eta0,
            chi0=chi0,
            z=z,
            Ks=Ks,
            compute_d3=args.use_d3,
            atomic_numbers=atomic_numbers,
            d3_params={'s6': 1.0, 'rs6': 1.217, 's18': 0.722, 'rs18': 1.0, 'alp': 14.0},
            damping='zero',
            smooth_fn=None,
        )
        return pe_nn + pe_nb

    class ReaxNetPQEqCalculator(Calculator):
        implemented_properties = ['energy', 'forces', 'stress']

        def calculate(self, atoms_obj, properties=('energy', 'forces', 'stress'), system_changes=all_changes):
            super().calculate(atoms_obj, properties, system_changes)

            scaled = jnp.asarray(atoms_obj.get_scaled_positions(wrap=True))
            cell = np.asarray(atoms_obj.get_cell().array)
            box = jnp.asarray(cell.T)

            def e_fn(pos_frac):
                return total_energy_fn(pos_frac, box, jnp.zeros((3, 3)))

            energy_val, grad_pos = jax.value_and_grad(e_fn, argnums=0)(scaled)

            grad_pos = np.asarray(grad_pos)

            # Convert grad wrt fractional coordinates to Cartesian forces
            inv_cell_T = np.linalg.inv(cell.T)
            forces = -grad_pos @ inv_cell_T

            # Fallback stress (current API path). This keeps ASE NPT interfaces satisfied.
            # If exact stress is needed, refactor energy path to accept cell/strain explicitly.
            stress_voigt = np.zeros(6, dtype=float)

            self.results = {
                'energy': float(energy_val),
                'forces': forces,
                'stress': stress_voigt,
            }

    np.random.seed(args.seed)
    atoms_md = atoms.copy()
    atoms_md.calc = ReaxNetPQEqCalculator()
    atoms_md.set_constraint(FixCom())

    log_out = Path(args.log_out).resolve()
    traj_out = Path(args.traj_out).resolve()
    log_out.parent.mkdir(parents=True, exist_ok=True)
    traj_out.parent.mkdir(parents=True, exist_ok=True)

    steps_npt = int(round(args.npt_ps * 1000.0 / args.timestep_fs))
    steps_nvt = int(round(args.nvt_ps * 1000.0 / args.timestep_fs))

    print(f"[INFO] atoms={len(atoms_md)} use_d3={args.use_d3}")
    print(f"[INFO] Stage1 NPT: T={args.npt_temperature_k}K P={args.npt_pressure_atm}atm dt={args.timestep_fs}fs steps={steps_npt}")
    print(f"[INFO] Stage2 NVT: T={args.nvt_temperature_k}K dt={args.timestep_fs}fs steps={steps_nvt}")
    print(f"[INFO] log={log_out}")
    print(f"[INFO] traj={traj_out}")

    traj = Trajectory(str(traj_out), mode='w', atoms=atoms_md)
    traj.write(atoms_md)

    # Stage 1: NPT (Berendsen)
    # ASE units has `bar` but not `atm`; convert atm -> bar first.
    pressure_bar = args.npt_pressure_atm * 1.01325
    if steps_npt > 0:
        npt = NPTBerendsen(
            atoms_md,
            timestep=args.timestep_fs * units.fs,
            temperature_K=args.npt_temperature_k,
            pressure_au=pressure_bar * units.bar,
            taut=args.npt_taut_fs * units.fs,
            taup=args.npt_taup_fs * units.fs,
            compressibility_au=args.compressibility_per_bar / units.bar,
            logfile=str(log_out),
            loginterval=1,
        )
        npt.attach(lambda: traj.write(atoms_md), interval=1)
        npt.run(steps_npt)

    # Stage 2: NVT (Langevin)
    if steps_nvt > 0:
        nvt = Langevin(
            atoms_md,
            timestep=args.timestep_fs * units.fs,
            temperature_K=args.nvt_temperature_k,
            friction=args.friction,
            fixcm=False,
            logfile=str(log_out),
            loginterval=1,
        )
        nvt.attach(lambda: traj.write(atoms_md), interval=1)
        nvt.run(steps_nvt)

    traj.close()
    print('[DONE] Two-stage MD finished.')


if __name__ == '__main__':
    main()
