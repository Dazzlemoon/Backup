import jax
import jax.numpy as jnp

from marathon.utils import masked


def get_loss_fn(predict_fn, weights={"energy": 1.0, "forces": 1.0}, correct_mean=False):
    """Get a loss function.

    A loss function is something that ingests a Batch and returns
    a MSE loss (for optimisation) and summed residuals (for other metrics).

    The assumption is that this can be vmapped or scanned across a whole
    bunch of batches at once.

    Hardcoded decisions for now:
        - Energy & stress loss is always scaled by number of atoms.
        - We take all the means over flattened data, i.e. each
            atom contributes with the same weight, as opposed to
            averaging over structures (graphs) first, which would
            weigh smaller structures higher.
        - We expect the loss weights to take care of variance scaling.

    """

    def loss_fn(params, batch):
        _, num_nodes_by_graph = jnp.unique(
            batch.node_to_graph, size=batch.graph_mask.shape[0], return_counts=True
        )
        inverse_num_nodes_by_graph = masked(
            lambda x: 1.0 / x,
            num_nodes_by_graph[:, None],
            batch.graph_mask,
            fn_value=1e6,
        ).flatten()

        predictions = predict_fn(params, batch)

        residuals = {
            key: predictions[key] - batch.labels[key] for key in predictions.keys()
        }

        if "energy" in residuals:
            residuals["energy"] = residuals["energy"] * inverse_num_nodes_by_graph
        if "stress" in residuals:
            residuals["stress"] = (
                residuals["stress"] * inverse_num_nodes_by_graph[..., None, None]
            )

        loss = jnp.array(0.0)
        for key, weight in weights.items():
            se = jax.lax.square(residuals[key]) * batch.labels[key + "_mask"]
            if correct_mean:
                summed = se.sum()
                factor = batch.labels[key + "_mask"].sum()
                factor = jnp.clip(factor, min=1.0)
                loss += weight * (summed / factor)
            else:
                loss += weight * jnp.mean(se)

        aux = {}
        for key, residual in residuals.items():
            mask = batch.labels[key + "_mask"]

            aux[f"{key}_abs"] = jnp.abs(residual * mask).sum(axis=0)
            aux[f"{key}_sq"] = jax.lax.square(residual * mask).sum(axis=0)

            # we need to count samples. so we reshape the mask to
            # [samples, flattened components] to give us the "real" samples
            mask = batch.labels[key + "_mask"]
            mask = mask.reshape(mask.shape[0], -1)
            aux[f"{key}_n"] = jnp.sum(mask.all(axis=1))

        return loss, aux

    return loss_fn
