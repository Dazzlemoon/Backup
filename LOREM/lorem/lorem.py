import jax
import jax.numpy as jnp
import numpy as np

from flax.core import FrozenDict
import flax.linen as nn
import e3x

import functools
from collections.abc import Sequence

from jaxpme import Ewald

from marathon.utils import masked
from sog_kernel import compute_sog_periodic_potentials


class Lorem(nn.Module):
    cutoff: float = 5.0
    max_degree: int = 6
    max_degree_lr: int = 2
    num_features: int = 128
    num_radial: int = 32
    num_species: int = 8
    num_spherical_features: int = 8
    cutoff_fn: str = "cosine_cutoff"
    radial_basis: str = "basic_bernstein"
    lr: bool = True
    lr_kernel_type: str = "coulomb"  # "coulomb" or "sog"
    sog_num_gaussians: int = 12
    sog_init_mode: str = "uniform"  # "uniform" or "dimer_cc"
    sog_l_dependent_params: bool = False
    num_message_passing: int = 0
    equivariant_message_passing: bool = False
    initialize_node_features: bool = False

    @staticmethod
    def _masked_mean(values, mask):
        mask_f = mask.astype(values.dtype)
        denom = jnp.maximum(mask_f.sum(), 1.0)
        return jnp.sum(values * mask_f) / denom

    @staticmethod
    def _inv_softplus(x):
        # Numerically stable inverse of softplus for x > 0.
        return jnp.log(jnp.expm1(x))

    @nn.compact
    def __call__(
        self,
        R_ij,
        i,
        j,
        Z_i,
        pair_mask,
        node_mask,
        # inputs for Ewald (periodic) or None (non-periodic)
        # if Ewald is used, the batch may only contain one real sample
        positions,
        cell,
        k_grid,  # only .shape matters (see jax-pme)
        smearing,
        # all-to-all edges (non-periodic) or None (periodic)
        full_R_ij,
        full_i,
        full_j,
        full_edge_mask,
    ):
        num_nodes = Z_i.shape[0]
        num_pairs = R_ij.shape[0]

        max_degree = self.max_degree
        max_degree_lr = self.max_degree_lr
        num_l = self.max_degree + 1
        num_lm = int((self.max_degree + 1) ** 2)

        d = self.num_features
        s = self.num_spherical_features

        # empirical factors to make var of equivariant norm more uniform across l
        l_factors = (
            jnp.array([(2 * l + 1) for l in range(max_degree + 1)], dtype=float) ** 0.25
        )

        # -- initial embeddings --
        radial, spherical, species, cutoffs, r_ij = Initial(
            cutoff=self.cutoff,
            max_degree=self.max_degree,
            num_features=self.num_features,
            num_radial=self.num_radial,
            num_species=self.num_species,
            num_spherical_features=self.num_spherical_features,
            cutoff_fn=self.cutoff_fn,
            radial_basis=self.radial_basis,
        )(
            R_ij,
            Z_i,
            pair_mask,
            node_mask,
        )

        # -- learned linear transformation of radial expansion --
        edges_scalar = RadialCoefficients(d)(
            jnp.concatenate([species[i], species[j]], axis=-1),
            radial,
            cutoffs,
            pair_mask,
        )

        # -- initial scalar and equivariant (spherical) node features
        if self.initialize_node_features:
            nodes_scalar = masked(nn.Dense(d, use_bias=True), species, node_mask)
        else:
            nodes_scalar = jnp.zeros((num_nodes, d), dtype=species.dtype)

        updates = (
            jax.ops.segment_sum(
                masked(nn.Dense(d, use_bias=False), edges_scalar, pair_mask),
                i,
                num_segments=num_nodes,
            )
            * node_mask[..., None]
        )
        nodes_scalar = Update(d)(nodes_scalar, updates, node_mask)

        coefficients = masked(
            nn.Dense(num_l * s, use_bias=False), edges_scalar, pair_mask
        ).reshape(num_pairs, num_l, s)
        coefficients = degree_wise_repeat_last_axis(coefficients, max_degree)
        edges_spherical = jnp.einsum("plf,pl->plf", coefficients, spherical)

        nodes_spherical = (
            jax.ops.segment_sum(
                edges_spherical.reshape(num_pairs, 1, num_lm, s),
                i,
                num_segments=num_nodes,
            )
            * node_mask[..., None, None, None]
        )
        nodes_spherical = e3x.nn.TensorDense(use_bias=False, include_pseudotensors=False)(
            nodes_spherical
        )

        # -- mix equivariant information into scalar node features --
        norms = spherical_norm_last_axis(nodes_spherical, max_degree)
        updates = (norms * l_factors[None, None, :, None]).reshape(num_nodes, -1)

        nodes_scalar = Update(d)(nodes_scalar, updates, node_mask)

        # -- initial prediction --
        energy_short = masked(MLP(features=[d, d, 1]), nodes_scalar, node_mask)[..., 0]

        # -- message passing (if turned on) --
        for _ in range(self.num_message_passing):
            edges_scalar = RadialCoefficients(d)(
                jnp.concatenate([nodes_scalar[i], nodes_scalar[j]], axis=-1),
                radial,
                cutoffs,
                pair_mask,
            )
            updates = (
                jax.ops.segment_sum(
                    masked(nn.Dense(d, use_bias=False), edges_scalar, pair_mask),
                    i,
                    num_segments=num_nodes,
                )
                * node_mask[..., None]
            )
            nodes_scalar = Update(d)(nodes_scalar, updates, node_mask)

            if self.equivariant_message_passing:
                coefficients = masked(
                    nn.Dense(num_l * s, use_bias=False), edges_scalar, pair_mask
                ).reshape(num_pairs, num_l, s)
                coefficients = degree_wise_repeat_last_axis(coefficients, max_degree)
                edges_spherical = jnp.einsum(
                    "plf,pl->plf", coefficients, spherical
                ).reshape(num_pairs, 1, num_lm, s)

                messages = (
                    e3x.nn.MessagePass(include_pseudotensors=False)(
                        nodes_spherical,
                        edges_spherical,
                        dst_idx=i,
                        src_idx=j,
                    )
                    * node_mask[..., None, None, None]
                )
                nodes_spherical = e3x.nn.Tensor(include_pseudotensors=False)(
                    e3x.nn.Dense(use_bias=False, features=s)(nodes_spherical),
                    e3x.nn.Dense(use_bias=False, features=s)(messages),
                )

                norms = spherical_norm_last_axis(nodes_spherical, max_degree)
                updates = (norms * l_factors[None, None, :, None]).reshape(num_nodes, -1)
                nodes_scalar = Update(d)(nodes_scalar, updates, node_mask)

            # -- residual prediction --
            energy_short += masked(MLP(features=[d, d, 1]), nodes_scalar, node_mask)[..., 0]

        scalar_charges = jnp.zeros((num_nodes, 1), dtype=nodes_scalar.dtype)
        spherical_charges = jnp.zeros((num_nodes, int((max_degree_lr + 1) ** 2)), dtype=nodes_scalar.dtype)
        energy_long = jnp.zeros_like(energy_short)
        if self.lr:
            # -- compute LR potentials --
            scalar_charges = masked(MLP(features=[2 * d, 1]), nodes_scalar, node_mask)
            spherical_charges = e3x.nn.TensorDense(
                features=1,
                use_bias=False,
                max_degree=max_degree_lr,
                include_pseudotensors=False,
            )(nodes_spherical).reshape(num_nodes, -1)
            charges = jnp.concatenate([scalar_charges, spherical_charges], axis=-1)

            if k_grid is not None:  # if periodic
                if self.lr_kernel_type == "sog":
                    if self.sog_init_mode == "dimer_cc":
                        if self.sog_num_gaussians != 12:
                            raise ValueError(
                                f"sog_init_mode='dimer_cc' requires sog_num_gaussians=12, got {self.sog_num_gaussians}"
                            )
                        # From CACE-SOG-Ji dimer-CC hardcoded init:
                        # shift_1 in [-3, 2], amplitude_1 fixed vector.
                        shift_1 = jnp.linspace(-3.0, 2.0, self.sog_num_gaussians)
                        widths = jnp.exp(2.0 * shift_1)  # width in exp(-k^2 * width)
                        sog_log_widths_init = Lorem._inv_softplus(widths)
                        sog_amplitudes_init = jnp.array(
                            [
                                -7.0450,
                                11.4645,
                                -4.9724,
                                0.4311,
                                0.1973,
                                -0.1282,
                                0.4223,
                                1.3309,
                                3.2130,
                                8.1743,
                                19.3299,
                                55.2736,
                            ]
                        )
                    else:
                        sog_log_widths_init = jnp.linspace(-2.0, 1.0, self.sog_num_gaussians)
                        sog_amplitudes_init = jnp.ones((self.sog_num_gaussians,))
                    if self.sog_l_dependent_params:
                        # Build channel-wise kernels grouped by multipole order l.
                        # Channel layout:
                        #   0: scalar charge;
                        #   1..: spherical channels grouped as l=0,1,...,max_degree_lr.
                        num_channels = charges.shape[-1]
                        l_per_channel = [0, 0]
                        for l in range(1, max_degree_lr + 1):
                            l_per_channel.extend([l] * (2 * l + 1))
                        if len(l_per_channel) != num_channels:
                            raise ValueError(
                                "Mismatch in SOG channel/l mapping: "
                                f"expected {num_channels} channels, got mapping length {len(l_per_channel)}"
                            )
                        l_per_channel = jnp.array(l_per_channel, dtype=jnp.int32)

                        sog_log_widths_per_l = self.param(
                            "sog_log_widths_per_l",
                            lambda key: jnp.tile(
                                sog_log_widths_init[None, :], (max_degree_lr + 1, 1)
                            ),
                        )
                        sog_amplitudes_per_l = self.param(
                            "sog_amplitudes_per_l",
                            lambda key: jnp.tile(
                                sog_amplitudes_init[None, :], (max_degree_lr + 1, 1)
                            ),
                        )
                        sog_log_widths = sog_log_widths_per_l[l_per_channel]
                        sog_amplitudes = sog_amplitudes_per_l[l_per_channel]
                        self.sow(
                            "intermediates",
                            "diag_sog_log_widths_per_l",
                            sog_log_widths_per_l,
                        )
                        self.sow(
                            "intermediates",
                            "diag_sog_amplitudes_per_l",
                            sog_amplitudes_per_l,
                        )
                    else:
                        sog_log_widths = self.param(
                            "sog_log_widths",
                            lambda key: sog_log_widths_init,
                        )
                        sog_amplitudes = self.param(
                            "sog_amplitudes",
                            lambda key: sog_amplitudes_init,
                        )
                    potentials = compute_sog_periodic_potentials(
                        charges=charges,
                        positions=positions,
                        cell=cell,
                        k_grid_shape=k_grid.shape,
                        sog_log_widths=sog_log_widths,
                        sog_amplitudes=sog_amplitudes,
                    )
                    self.sow("intermediates", "diag_sog_amplitudes", sog_amplitudes)
                    self.sow("intermediates", "diag_sog_log_widths", sog_log_widths)
                else:
                    calculator = Ewald(full_neighbor_list=True)
                    potentials = jax.vmap(
                        lambda q: calculator.potentials(
                            q,
                            cell,
                            positions,
                            i,
                            j,
                            None,
                            k_grid,
                            smearing,
                            atom_mask=node_mask,
                            pair_mask=pair_mask,
                            distances=r_ij,
                        ),
                        in_axes=-1,
                        out_axes=-1,
                    )(charges)
            elif full_R_ij is not None:  # if non-periodic
                full_r_ij = e3x.ops.norm(full_R_ij, axis=-1)
                mask = full_r_ij == 0
                masked_r_ij = jnp.where(mask, 1e-6, full_r_ij)
                one_over_r = jnp.where(mask, 0.0, 1 / masked_r_ij)
                potentials = jax.ops.segment_sum(
                    charges[full_j] * one_over_r[..., None], full_i, num_segments=num_nodes
                )

            scalar_potential = potentials[..., 0][..., None]
            spherical_potential = potentials[..., 1:].reshape(num_nodes, 1, -1, 1)

            # -- combine LR potentials back into local features --
            spherical_potential = e3x.nn.Dense(s, use_bias=False)(spherical_potential)
            spherical_updates = e3x.nn.Tensor(include_pseudotensors=False)(
                spherical_potential, nodes_spherical
            )

            norms = spherical_norm_last_axis(spherical_updates, max_degree)
            norms = (norms * l_factors[None, None, :, None]).reshape(num_nodes, -1)
            updates = jnp.concatenate([scalar_potential, norms], axis=-1)
            nodes_scalar = Update(d)(nodes_scalar, updates, node_mask)

            # -- residual prediction --
            energy_long = masked(MLP(features=[d, d, 1]), nodes_scalar, node_mask)[..., 0]

        energy = energy_short + energy_long

        # Diagnostics are collected only when `intermediates` is mutable.
        self.sow(
            "intermediates",
            "diag_scalar_charge_mean",
            Lorem._masked_mean(scalar_charges[..., 0], node_mask),
        )
        self.sow(
            "intermediates",
            "diag_scalar_charge_abs_mean",
            Lorem._masked_mean(jnp.abs(scalar_charges[..., 0]), node_mask),
        )
        self.sow(
            "intermediates",
            "diag_spherical_charge_abs_mean",
            Lorem._masked_mean(jnp.mean(jnp.abs(spherical_charges), axis=-1), node_mask),
        )
        self.sow(
            "intermediates",
            "diag_energy_short_mean",
            Lorem._masked_mean(energy_short, node_mask),
        )
        self.sow(
            "intermediates",
            "diag_energy_long_mean",
            Lorem._masked_mean(energy_long, node_mask),
        )
        self.sow(
            "intermediates",
            "diag_energy_total_mean",
            Lorem._masked_mean(energy, node_mask),
        )
        self.sow("intermediates", "diag_scalar_charges_raw", scalar_charges)
        self.sow("intermediates", "diag_spherical_charges_raw", spherical_charges)
        self.sow("intermediates", "diag_energy_short_raw", energy_short)
        self.sow("intermediates", "diag_energy_long_raw", energy_long)

        return energy

    def dummy_inputs(self, dtype=jnp.float32):
        return (
            jnp.array([[0, 0, 0], [1, 1, 1], [0.5, 1, 1], [0, 1, 0]], dtype=dtype),
            jnp.array([0, 1, 2, 2]),
            jnp.array([1, 0, 2, 2]),
            jnp.array([0, 0, 0]),
            jnp.array([True, True, False, False]),
            jnp.array([True, True, False]),
            jnp.array([[0, 0, 0], [1, 1, 1], [0.5, 1, 1]], dtype=dtype),
            jnp.eye(3),
            jnp.ones((4, 4, 4)),
            jnp.array(1.0),
            jnp.array([[0, 0, 0], [1, 1, 1], [0.5, 1, 1]], dtype=dtype),
            jnp.array([0, 1, 2]),
            jnp.array([1, 0, 2]),
            jnp.array([True, True, False]),
        )

    def energy(self, params, batch):
        energies = self.apply(
            params,
            batch.edges,
            batch.centers,
            batch.others,
            batch.nodes,
            batch.edge_mask,
            batch.node_mask,
            batch.positions,
            batch.cell,
            batch.k_grid,
            batch.smearing,
            batch.full_edges,
            batch.full_centers,
            batch.full_others,
            batch.full_edge_mask,
        )
        energies *= batch.node_mask

        return jnp.sum(energies), energies

    def predict(self, params, batch, stress=False):
        energy_and_derivatives_fn = jax.value_and_grad(
            self.energy, allow_int=True, has_aux=True, argnums=1
        )
        batch_energy_and_atom_energies, grads = energy_and_derivatives_fn(params, batch)
        _, energies = batch_energy_and_atom_energies

        energy = jax.ops.segment_sum(
            energies, batch.node_to_graph, batch.graph_mask.shape[0]
        )

        dR_ij = grads.edges * batch.edge_mask[..., None]
        forces_1 = jax.ops.segment_sum(
            dR_ij, batch.centers, batch.nodes.shape[0], indices_are_sorted=False
        )
        forces_2 = jax.ops.segment_sum(
            dR_ij, batch.others, batch.nodes.shape[0], indices_are_sorted=False
        )

        forces = (forces_1 - forces_2) * batch.node_mask[..., None]

        if batch.positions is not None:
            forces_3 = -grads.positions * batch.node_mask[..., None]

            forces += forces_3 * batch.node_mask[..., None]

        elif batch.full_edges is not None:
            full_dR_ij = grads.full_edges * batch.full_edge_mask[..., None]
            forces_3 = jax.ops.segment_sum(
                full_dR_ij,
                batch.full_centers,
                batch.nodes.shape[0],
                indices_are_sorted=False,
            )
            forces_4 = jax.ops.segment_sum(
                full_dR_ij,
                batch.full_others,
                batch.nodes.shape[0],
                indices_are_sorted=False,
            )

            forces += (forces_3 - forces_4) * batch.node_mask[..., None]

        if batch.positions is not None and batch.full_edges is not None:
            raise ValueError

        results = {"energy": energy, "forces": forces}

        return results


# -- initial embeddings --


class Initial(nn.Module):
    cutoff: float = 5.0
    max_degree: int = 4
    num_features: int = 128
    num_radial: int = 32
    num_species: int = 8
    num_spherical_features: int = 4
    cutoff_fn: str = "cosine_cutoff"
    radial_basis: str = "basic_bernstein"

    @nn.compact
    def __call__(
        self,
        R_ij,
        Z_i,
        pair_mask,
        node_mask,
    ):
        cutoff_fn = getattr(e3x.nn.functions, self.cutoff_fn)

        R_ij, r_ij = e3x.ops.normalize_and_return_norm(R_ij, axis=-1)
        R_ij *= pair_mask[..., None]

        cutoffs = cutoff_fn(r_ij, cutoff=self.cutoff) * pair_mask  # -> [pairs]

        radial_expansion = (
            RadialEmbedding(
                self.num_radial,
                self.cutoff,
                function=self.radial_basis,
            )(r_ij)
            * cutoffs[..., None]
        )

        spherical_expansion = e3x.so3.spherical_harmonics(
            R_ij, self.max_degree, r_is_normalized=True
        )
        spherical_expansion *= pair_mask[..., None]

        species_expansion = (
            ChemicalEmbedding(num_features=self.num_species)(Z_i) * node_mask[..., None]
        )

        return radial_expansion, spherical_expansion, species_expansion, cutoffs, r_ij


class ChemicalEmbedding(nn.Module):
    num_features: int
    total_species: int = 100

    @nn.compact
    def __call__(self, species):
        return nn.Embed(num_embeddings=self.total_species, features=self.num_features)(
            species
        )


class RadialEmbedding(nn.Module):
    num_features: int
    cutoff: int
    function: str = "basic_gaussian"
    args: FrozenDict = FrozenDict({})
    learned_transform: bool = False

    @nn.compact
    def __call__(self, r):
        function = getattr(e3x.nn.functions, self.function)

        expansion = function(
            r, **{"limit": self.cutoff, "num": self.num_features, **self.args}
        )

        if self.learned_transform:
            expansion = nn.Dense(features=self.num_features, use_bias=False)(expansion)

        return expansion


# -- basic modules --


class MLP(nn.Module):
    features: Sequence[int]
    activation: str = "silu"
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        activation = getattr(jax.nn, self.activation)
        num_layers = len(self.features)

        for i, f in enumerate(self.features):
            x = nn.Dense(features=f, use_bias=self.use_bias)(x)
            if i != num_layers - 1:
                x = activation(x)

        return x


class Update(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, y, node_mask):
        x += masked(
            MLP(features=[2 * self.features, self.features]),
            y,
            node_mask,
        )
        x = masked(nn.LayerNorm(), x, node_mask)
        x += masked(MLP(features=[2 * self.features, self.features]), x, node_mask)
        x = masked(nn.LayerNorm(), x, node_mask)

        return x


# -- other modules --


class RadialCoefficients(nn.Module):
    features: int

    @nn.compact
    def __call__(self, pair_features, radial_expansion, cutoffs, pair_mask):
        num_radial = radial_expansion.shape[-1]

        coefficients = masked(
            MLP(
                features=[
                    self.features,
                    num_radial * self.features,
                ]
            ),
            pair_features,
            pair_mask,
        )
        coefficients = coefficients.reshape(-1, num_radial, self.features)
        coefficients = jnp.einsum("prf,pr->pf", coefficients, radial_expansion)

        return coefficients


# -- helpers to deal with spherical features --


def degree_wise_trace(
    x,
    max_degree,
):
    segments = np.concatenate(
        [np.array([l] * (2 * l + 1)) for l in range(max_degree + 1)]
    ).reshape(-1)

    return jax.vmap(
        lambda _x: jax.ops.segment_sum(_x, segments, num_segments=(max_degree + 1)),
    )(x)


def degree_wise_repeat(x, max_degree, axis):
    repeats = np.array([2 * l + 1 for l in range(max_degree + 1)])

    return jnp.repeat(x, repeats, total_repeat_length=repeats.sum(), axis=axis)


def degree_wise_repeat_last_axis(x, max_degree: int):
    return jax.vmap(
        lambda y: degree_wise_repeat(y, max_degree, -1), in_axes=-1, out_axes=-1
    )(x)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def spherical_norm(X, max_degree):
    squared = jax.lax.square(X)
    trace = degree_wise_trace(squared, max_degree)
    norm = jnp.sqrt(trace)
    return norm


@spherical_norm.defjvp
def spherical_norm_jvp(max_degree, primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = spherical_norm(x, max_degree)

    x_hat = x / degree_wise_repeat(jnp.where(primal_out > 0, primal_out, 1), max_degree, -1)

    tangent_out = degree_wise_trace(x_dot * x_hat, max_degree)
    return primal_out, tangent_out


def spherical_norm_last_axis(X, max_degree):
    # X is a e3x-style array, i.e. [batch, 1|2, lm, features]:
    # we vmap over parity and feature dimensions
    return jax.vmap(
        lambda z: jax.vmap(
            lambda x: spherical_norm(x, max_degree), in_axes=-1, out_axes=-1
        )(z),
        in_axes=1,
        out_axes=1,
    )(X)
