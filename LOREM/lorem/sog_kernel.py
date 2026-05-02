import jax
import jax.numpy as jnp


def _build_k_vectors(cell: jnp.ndarray, k_grid_shape):
    """Build reciprocal-space vectors from k-grid shape."""
    nx, ny, nz = [int(v) for v in k_grid_shape]
    gx = jnp.arange(nx, dtype=cell.dtype) - (nx // 2)
    gy = jnp.arange(ny, dtype=cell.dtype) - (ny // 2)
    gz = jnp.arange(nz, dtype=cell.dtype) - (nz // 2)
    grid = jnp.stack(jnp.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    recip = 2.0 * jnp.pi * jnp.linalg.inv(cell).T
    return grid @ recip, grid


def compute_sog_periodic_potentials(
    charges: jnp.ndarray,
    positions: jnp.ndarray,
    cell: jnp.ndarray,
    k_grid_shape,
    sog_log_widths: jnp.ndarray,
    sog_amplitudes: jnp.ndarray,
) -> jnp.ndarray:
    """
    Periodic long-range potentials via SOG Fourier multiplier.

    Returns:
        potentials with shape [num_nodes, num_channels]
    """
    kvec, integer_grid = _build_k_vectors(cell, k_grid_shape)
    k_sq = jnp.sum(kvec**2, axis=-1)  # [K]
    # Stable positive widths for Gaussian basis.
    widths = jax.nn.softplus(sog_log_widths) + 1e-8
    # Shared-kernel mode: sog_* shape [M]
    # Channel-specific mode: sog_* shape [C, M]
    if widths.ndim == 1:
        multiplier = jnp.sum(
            sog_amplitudes[None, :] * jnp.exp(-k_sq[:, None] * widths[None, :]),
            axis=-1,
        )  # [K]
    elif widths.ndim == 2:
        multiplier = jnp.sum(
            sog_amplitudes[None, :, :] * jnp.exp(-k_sq[:, None, None] * widths[None, :, :]),
            axis=-1,
        )  # [K, C]
    else:
        raise ValueError(
            f"Expected sog_log_widths ndim in {{1,2}}, got {widths.ndim}"
        )
    # Remove zero mode for stability and compatibility with Ewald-like behavior.
    is_zero_mode = jnp.all(integer_grid == 0, axis=-1)
    if multiplier.ndim == 1:
        multiplier = jnp.where(is_zero_mode, 0.0, multiplier)
    else:
        multiplier = jnp.where(is_zero_mode[:, None], 0.0, multiplier)

    phase = positions @ kvec.T  # [N, K]
    exp_minus = jnp.exp(-1j * phase)  # [N, K]
    exp_plus = jnp.exp(1j * phase)  # [N, K]
    structure = jnp.einsum("nk,nc->kc", exp_minus, charges)  # [K, C]
    if multiplier.ndim == 1:
        weighted = multiplier[:, None] * structure  # [K, C]
    else:
        weighted = multiplier * structure  # [K, C]
    volume = jnp.maximum(jnp.abs(jnp.linalg.det(cell)), 1e-12)
    potentials_complex = jnp.einsum("nk,kc->nc", exp_plus, weighted) / volume
    return jnp.real(potentials_complex)
