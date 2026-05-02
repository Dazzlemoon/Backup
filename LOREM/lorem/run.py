if __name__ == "__main__":
    from marathon.io import read_yaml

    settings = read_yaml("settings.yaml")

    # -- settings --

    from marathon.data import datasets

    data_train = datasets / settings["train"]
    data_valid = datasets / settings["valid"]

    # {name: (source, save_predictions)}
    test_datasets = {
        "valid": (datasets / settings["valid"], False),
    }
    if "test_datasets" in settings:
        for k, (data, save) in settings["test_datasets"].items():
            test_datasets[k] = (datasets / data, save)

    # only support batch_shape for now
    batch_style = "batch_shape"
    # for the periodic case, chunk_size takes the role of num_graphs,
    # we vmap over chunks and accumulate gradients
    chunk_size = settings.get("chunk_size", 1)

    num_graphs = settings["num_graphs"]  # must be 2 for periodic
    num_nodes = settings["num_nodes"]
    num_edges = settings["num_edges"]

    loss_weights = settings.get("loss_weights", {"energy": 0.5, "forces": 0.5})
    scale_by_variance = settings.get("scale_by_variance", False)

    start_learning_rate = float(settings.get("start_learning_rate", 1e-3))
    min_learning_rate = float(settings.get("min_learning_rate", 1e-6))

    max_epochs = settings.get("max_epochs", 2000)
    valid_every_epoch = settings.get("valid_every_epoch", 2)

    optimizer = settings.get("optimizer", "lamb")

    # lr decay
    decay_style = settings.get("decay_style", "linear")
    start_decay_after = settings.get("start_decay_after", 10)
    stop_decay_after = settings.get(
        "stop_decay_after", max_epochs
    )  # ignored for exponential

    seed = settings.get("seed", 0)
    print_model_summary = True
    benchmark_pipeline = settings.get("benchmark_pipeline", True)
    workdir = "run"

    use_wandb = settings.get("use_wandb", True)
    # used for wandb -- use folder names by default
    wandb_project = None
    wandb_name = None

    default_matmul_precision = settings.get("default_matmul_precision", "default")
    debug_nans = settings.get("debug_nans", False)  # ~50% slowdown, use with care
    enable_x64 = settings.get("enable_x64", False)
    correct_mean = settings.get("correct_mean", False)
    lr_kernel_type = settings.get("lr_kernel_type", "coulomb")
    sog_num_gaussians = int(settings.get("sog_num_gaussians", 12))
    sog_init_mode = settings.get("sog_init_mode", None)
    sog_l_dependent_params = bool(settings.get("sog_l_dependent_params", False))

    # settings for grain
    worker_count = settings.get("worker_count", 4)
    worker_buffer_size = settings.get("worker_buffer_size", 2)

    # -- imports & startup --

    import numpy as np
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", enable_x64)
    jax.config.update("jax_default_matmul_precision", default_matmul_precision)
    jax.config.update("jax_debug_nans", debug_nans)

    from pathlib import Path

    from marathon import comms

    reporter = comms.reporter()
    reporter.start("run")
    reporter.step("startup")

    # -- housekeeping based on settings --
    keys = list(loss_weights.keys())
    use_stress = "stress" in keys

    workdir = Path(workdir)

    # -- randomness --
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)

    # -- model --
    from marathon.io import from_dict, read_yaml

    model_config = read_yaml("model.yaml")
    assert "model" in model_config
    assert "baseline" in model_config  # for compatibility w/ external models
    species_to_weight = model_config["baseline"]["elemental"]

    if lr_kernel_type not in ("coulomb", "sog"):
        raise ValueError(
            f"Unsupported lr_kernel_type={lr_kernel_type!r}. Expected 'coulomb' or 'sog'."
        )
    # Keep model spec schema valid: from_dict expects `model` to contain only
    # model spec entries (e.g. {"lorem.Lorem": {...}}), not extra top-level keys.
    # Therefore inject kernel options into the inner model args dict.
    model_spec = model_config["model"]
    if "lorem.Lorem" in model_spec:
        model_key = "lorem.Lorem"
    elif len(model_spec) == 1:
        model_key = next(iter(model_spec.keys()))
    else:
        raise ValueError(
            "Could not infer model key for kernel injection. "
            f"model spec keys={list(model_spec.keys())}"
        )

    model_spec[model_key]["lr_kernel_type"] = lr_kernel_type
    model_spec[model_key]["sog_num_gaussians"] = sog_num_gaussians
    if sog_init_mode is not None:
        model_spec[model_key]["sog_init_mode"] = sog_init_mode
    model_spec[model_key]["sog_l_dependent_params"] = sog_l_dependent_params
    comms.talk(
        f"Long-range kernel selection: lr_kernel_type={lr_kernel_type}, "
        f"sog_num_gaussians={sog_num_gaussians}, "
        f"sog_l_dependent_params={sog_l_dependent_params}"
    )

    model = from_dict(model_config["model"])
    cutoff = model.cutoff

    params = model.init(init_key, *model.dummy_inputs())

    if print_model_summary:
        from flax import linen as nn

        msg = nn.tabulate(model, init_key)(*model.dummy_inputs())
        comms.state(msg.split("\n"), title="Model Summary")

    num_parameters = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    comms.state(f"Parameter count: {num_parameters}")

    # -- checkpointers --
    from marathon.emit import SummedMetric

    checkpointers = []

    which_checkpointers = settings.get("checkpointers", "default")

    name = "R2_" + "+".join([k[0].upper() for k in keys])
    checkpointers.append(SummedMetric(name, "r2", keys=keys))

    name = "MAE_" + "+".join([k[0].upper() for k in ["forces"]])
    checkpointers.append(SummedMetric(name, "mae", keys=["forces"]))

    if which_checkpointers == "full":
        name = "MAE_" + "+".join([k[0].upper() for k in ["energy"]])
        checkpointers.append(SummedMetric(name, "mae", keys=["energy"]))

        name = "RMSE_" + "+".join([k[0].upper() for k in ["energy"]])
        checkpointers.append(SummedMetric(name, "rmse", keys=["energy"]))

        name = "RMSE_" + "+".join([k[0].upper() for k in ["forces"]])
        checkpointers.append(SummedMetric(name, "rmse", keys=["forces"]))

    checkpointers = tuple(checkpointers)

    # -- data loading --
    from marathon.evaluate.metrics import get_stats
    from marathon.extra.hermes import (
        DataLoader,
        DataSource,
        FilterEmpty,
        IndexSampler,
        ToStack,
        prefetch_to_device,
    )
    from marathon.extra.hermes.pain import Record, RecordMetadata
    from transforms import ToFixedShapeBatch, SetUpEwald, ToSample

    to_sample = ToSample(cutoff=cutoff, energy=True, forces=True, stress=use_stress)
    prepare_ewald = SetUpEwald(lr_wavelength=cutoff / 8, smearing=cutoff / 4)

    def get_batcher():
        assert batch_style == "batch_shape"
        return ToFixedShapeBatch(
            num_graphs=num_graphs, num_edges=num_edges, num_nodes=num_nodes
        )

    source_train = DataSource(data_train, species_to_weight=species_to_weight)
    source_valid = DataSource(data_valid, species_to_weight=species_to_weight)
    baseline = {"elemental": species_to_weight}
    n_train = len(source_train)
    n_valid = len(source_valid)

    max_steps = max_epochs * n_train
    valid_every = valid_every_epoch * n_train
    comms.talk(f"run for {max_epochs} epochs, {max_steps} steps", full=True)
    comms.talk(
        f"validate every {valid_every_epoch} epochs, every {valid_every} steps",
        full=True,
    )

    reporter.step("loading validation set")

    # for now we assume that validation set fits into RAM easily
    valid_samples = []
    batcher = get_batcher()

    def valid_iterator():
        filterer = FilterEmpty()
        for i in range(n_valid):
            sample = to_sample.map(source_valid[i])
            if filterer.filter(sample):
                valid_samples.append(sample)
                yield Record(data=sample, metadata=RecordMetadata(index=i, record_key=i))

    data_valid = [prepare_ewald.map(b.data) for b in batcher(valid_iterator())]
    valid_stats = get_stats(valid_samples, keys=keys)

    valid_batch_sizes = np.array([batch.graph_mask.sum() for batch in data_valid])
    median_valid_batch_size = int(np.median(valid_batch_sizes))

    if scale_by_variance:
        old_loss_weights = loss_weights

        loss_weights = {k: v / valid_stats[k]["var"] for k, v in loss_weights.items()}

        msg = []
        for k, v in loss_weights.items():
            msg.append(f"{k}: {old_loss_weights[k]:.3f} -> {v:.3f}")
        comms.state(msg, title="variance scaled loss weights")

    del valid_samples

    reporter.step("setup training pipeline")

    def get_training_iterator(num_epochs):
        batchers = [get_batcher(), prepare_ewald]

        if chunk_size > 1:
            batchers.append(ToStack(batch_size=chunk_size, drop_remainder=True))

        return iter(
            DataLoader(
                data_source=source_train,
                sampler=IndexSampler(
                    n_train,
                    num_epochs=num_epochs,
                    seed=seed,
                ),
                operations=[
                    to_sample,
                    FilterEmpty(),
                    *batchers,
                ],
                worker_count=worker_count,
                worker_buffer_size=worker_buffer_size,
            )
        )

    if benchmark_pipeline:
        from time import monotonic

        reporter.step("benchmark training pipeline", spin=False)

        @jax.jit
        def test_fn(batch):
            if chunk_size == 1:
                return (
                    batch.edge_mask.sum(),
                    batch.node_mask.sum(),
                    batch.graph_mask.sum(),
                    batch.edge_mask.shape[0],
                    batch.node_mask.shape[0],
                    batch.graph_mask.shape[0],
                )
            else:
                return (
                    batch.edge_mask.sum(),
                    batch.node_mask.sum(),
                    batch.graph_mask.sum(),
                    batch.edge_mask.shape[0] * batch.edge_mask.shape[1],
                    batch.node_mask.shape[0] * batch.node_mask.shape[1],
                    batch.graph_mask.shape[0] * batch.graph_mask.shape[1],
                )

        # trigger jit
        test_fn(next(get_training_iterator(1)))

        test_iter = prefetch_to_device(get_training_iterator(1), 2)

        results = []
        start = monotonic()
        for i, batch in enumerate(test_iter):
            reporter.tick(f"chunk {i}")
            results.append(test_fn(batch))
            del batch
        results = np.array(results)
        duration = monotonic() - start

        real_samples = results[:, 2].sum()
        util_edges = 100 * results[:, 0] / results[:, 3]
        util_nodes = 100 * results[:, 1] / results[:, 4]
        util_samples = 100 * results[:, 2] / results[:, 5]
        pipeline_speed = duration / real_samples

        unique_edges = np.unique(results[:, 3]).shape[0]
        unique_nodes = np.unique(results[:, 4]).shape[0]
        unique_samples = np.unique(results[:, 4]).shape[0]

        num_chunks = i + 1
        num_batches = num_chunks * chunk_size

        msg = []
        msg.append(f"speed       : {1e6*pipeline_speed:.0f}µs/sample")
        msg.append(f"              {worker_count} workers, buffer {worker_buffer_size}")
        msg.append(
            f"edges  : {np.mean(util_edges):.2f}% / {np.mean(results[:, 0]):.0f} mean"
        )
        msg.append(
            f"nodes  : {np.mean(util_nodes):.2f}% / {np.mean(results[:, 1]):.0f} mean"
        )
        msg.append(
            f"samples: {np.mean(util_samples):.2f}% / {np.mean(results[:, 2]):.0f} mean"
        )

        msg.append("")
        msg.append(
            f"unique shapes: {unique_edges} edges, {unique_nodes} nodes, {unique_samples} samples"
        )
        msg.append(
            f"... -> expecting {unique_edges*unique_nodes*unique_samples} compilations"
        )
        msg.append("")
        if chunk_size > 1:
            msg.append(f"num chunks: {num_chunks} containing {chunk_size} batches")
        msg.append(
            f"num batches: {num_batches} ({real_samples/num_batches:.0f} samples/batch)"
        )

        comms.state(msg, title="Training Pipeline Statistics")

        if np.mean(util_edges) < 50 or np.mean(util_nodes) < 50:
            comms.warn("Ratio of real to padded edges or nodes is TOO LOW (<50%). No!")
            comms.warn("I SHOULD REFUSE TO CONTINUE WITH THIS SICK JOB ...")

        median_train_batch_size = int(np.median(results[:, 2]) / chunk_size)

        median_batch_size = median_train_batch_size
        batches_per_epoch = num_batches
    else:
        pipeline_speed = 0.0
        median_batch_size = median_valid_batch_size
        batches_per_epoch = int(len(source_train) / median_batch_size)

    comms.talk(f"estimated samples/batch: {median_batch_size}")
    comms.talk(f"estimated batches/epoch: {batches_per_epoch}")

    iter_train = get_training_iterator(max_epochs)

    # -- optimizer --
    import optax

    reporter.step("setup optimizer")

    if decay_style == "linear":
        transition_steps = stop_decay_after * batches_per_epoch
        initial_steps = start_decay_after * batches_per_epoch
        scheduler = optax.schedules.linear_schedule(
            init_value=start_learning_rate,
            end_value=min_learning_rate,
            transition_begin=initial_steps,
            transition_steps=transition_steps - initial_steps,
        )

    elif decay_style == "exponential":
        transition_steps = max_epochs * batches_per_epoch
        initial_steps = start_decay_after * batches_per_epoch
        decay_rate = min_learning_rate / start_learning_rate
        scheduler = optax.schedules.exponential_decay(
            init_value=start_learning_rate,
            transition_steps=transition_steps - initial_steps,
            transition_begin=initial_steps,
            decay_rate=decay_rate,
            end_value=min_learning_rate,
        )

    opt = getattr(optax, optimizer)

    @optax.inject_hyperparams
    def optimizer(learning_rate):
        return opt(learning_rate)

    optimizer = optimizer(scheduler)

    initial_opt_state = optimizer.init(params)

    # -- assemble state / handle restore --

    state = {
        "step": 0,
        "checkpointers": checkpointers,
        "opt_state": initial_opt_state,
        "iter_train": iter_train.get_state(),
    }

    if workdir.is_dir():
        from marathon.emit import get_latest

        comms.warn(
            f"found working directory {workdir}, will restore (only) model and optimisation state!"
        )
        reporter.step("restoring")

        items = get_latest(workdir, state)

        if items is None:
            comms.warn(f"failed to find checkpoints in workdir {workdir}, ignoring")
        else:
            params, state, new_model, _, _, _ = items

            comms.talk(f"restored step {state['step']}")

            # try to catch the most obvious error: editing the model config between restarts
            from marathon.io import to_dict

            assert to_dict(new_model) == to_dict(model)

            iter_train.set_state(state["iter_train"])
    else:
        workdir.mkdir()

    opt_state = state["opt_state"]

    # -- loggers --
    from marathon.io import to_dict

    from marathon.emit import Txt

    reporter.step("setup loggers")

    training_pipeline = {
        "style": "shape",
        "num_graphs": num_graphs,
        "num_edges": num_edges,
        "num_nodes": num_nodes,
    }

    if decay_style == "linear":
        lr_decay = {
            "style": "linear",
            "start_decay_after": start_decay_after,
            "stop_decay_after": stop_decay_after,
        }
    elif decay_style == "exponential":
        lr_decay = {"style": "exponential", "start_decay_after": start_decay_after}
    else:
        raise ValueError

    config = {
        "n_train": n_train,
        "n_valid": n_valid,
        "loss_weights": loss_weights,
        "max_steps": max_steps,
        "start_learning_rate": start_learning_rate,
        "min_learning_rate": min_learning_rate,
        "lr_kernel_type": lr_kernel_type,
        "sog_num_gaussians": sog_num_gaussians,
        "sog_init_mode": sog_init_mode,
        "sog_l_dependent_params": sog_l_dependent_params,
        "lr_decay": lr_decay,
        "chunk_size": chunk_size,
        "training_pipeline": training_pipeline,
        "valid_every": valid_every,
        "model": to_dict(model),
        "num_parameters": num_parameters,
        "worker_count": worker_count,
        "worker_buffer_size": worker_buffer_size,
    }

    metrics = {key: ["r2", "mae", "rmse"] for key in keys}

    loggers = [Txt(metrics=metrics)]

    if use_wandb:
        import wandb

        from marathon.emit import WandB

        this_folder = Path.cwd()

        if wandb_project is None:
            wandb_project = f"{this_folder.parent.parent.stem}.{this_folder.parent.stem}"

        if wandb_name is None:
            wandb_name = f"{this_folder.stem}"

        run = wandb.init(config=config, name=wandb_name, project=wandb_project)

        config["wandb_id"] = run.id

        loggers.append(WandB(run, metrics=metrics))

    # -- setup actual training loop --
    from time import monotonic

    from marathon.emit import save_checkpoints
    from marathon.evaluate import get_loss_fn, get_metrics_fn, get_predict_fn
    from marathon.utils import s_to_string, tree_stack

    reporter.step("setup training loop")

    pred_fn = lambda params, batch: model.predict(params, batch, stress=use_stress)

    _loss_fn = get_loss_fn(pred_fn, weights=loss_weights, correct_mean=correct_mean)

    if chunk_size > 1:

        def loss_fn(params, batch):
            losses, auxs = jax.vmap(lambda x: _loss_fn(params, x))(batch)
            loss = losses.mean()
            aux = jax.tree.map(lambda x: x.sum(axis=0), auxs)

            return loss, aux

    else:
        loss_fn = _loss_fn

    loss_fn = jax.jit(loss_fn)

    train_metrics_fn = get_metrics_fn(keys=keys)  # no stats
    valid_metrics_fn = get_metrics_fn(keys=keys, stats=valid_stats)

    diag_keys = (
        "diag_scalar_charge_mean",
        "diag_scalar_charge_abs_mean",
        "diag_spherical_charge_abs_mean",
        "diag_energy_short_mean",
        "diag_energy_long_mean",
        "diag_energy_total_mean",
    )
    diag_raw_keys = (
        "diag_scalar_charges_raw",
        "diag_spherical_charges_raw",
        "diag_energy_short_raw",
        "diag_energy_long_raw",
    )

    @jax.jit
    def diagnostics_fn(params, batch):
        _, vars_out = model.apply(
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
            mutable=["intermediates"],
        )
        diagnostics = vars_out["intermediates"]
        return {k: diagnostics[k][0] for k in (*diag_keys, *diag_raw_keys)}

    def _forces_from_grads(grads, batch):
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
        return forces

    @jax.jit
    def force_ratio_fn(params, batch):
        # Compute ||F_long||/||F_total|| on the fly from gradients of
        # long-range and total energies, matching the CACE-LOREM diagnostic.
        def total_energy_fn(p, b):
            e_sum, _ = model.energy(p, b)
            return e_sum

        def long_energy_fn(p, b):
            _, vars_out = model.apply(
                p,
                b.edges,
                b.centers,
                b.others,
                b.nodes,
                b.edge_mask,
                b.node_mask,
                b.positions,
                b.cell,
                b.k_grid,
                b.smearing,
                b.full_edges,
                b.full_centers,
                b.full_others,
                b.full_edge_mask,
                mutable=["intermediates"],
            )
            d = vars_out["intermediates"]
            energy_long = d["diag_energy_long_raw"][0] * b.node_mask
            return jnp.sum(energy_long)

        grads_total = jax.grad(total_energy_fn, argnums=1, allow_int=True)(params, batch)
        grads_long = jax.grad(long_energy_fn, argnums=1, allow_int=True)(params, batch)

        forces_total = _forces_from_grads(grads_total, batch)
        forces_long = _forces_from_grads(grads_long, batch)

        eps = 1e-12
        long_norm = jnp.sum(jnp.linalg.norm(forces_long, axis=-1))
        total_norm = jnp.sum(jnp.linalg.norm(forces_total, axis=-1))
        return long_norm / jnp.maximum(total_norm, eps)

    def build_detailed_diag(diag_out, batch):
        graph_mask = np.asarray(batch.graph_mask).astype(bool)
        node_mask = np.asarray(batch.node_mask).astype(bool)
        node_to_graph = np.asarray(batch.node_to_graph)
        num_nodes_total = node_mask.shape[0]

        # `batch.positions` may be None (non-periodic pipeline), which becomes
        # a 0-d object array under np.asarray(...). Keep logging robust by
        # falling back to NaN coordinates in that case.
        positions = np.asarray(batch.positions)
        if positions.ndim != 2 or positions.shape[0] != num_nodes_total or positions.shape[1] != 3:
            positions = np.full((num_nodes_total, 3), np.nan, dtype=float)

        species = np.asarray(batch.nodes)
        if species.ndim > 1:
            # If nodes are one-hot / embeddings, use argmax as readable species id.
            species = np.argmax(species, axis=-1)

        real_graph_ids = np.where(graph_mask)[0]
        if real_graph_ids.shape[0] == 0:
            graph_id = 0
        else:
            graph_id = int(real_graph_ids[0])

        node_sel = node_mask & (node_to_graph == graph_id)
        scalar = np.asarray(diag_out["diag_scalar_charges_raw"])[node_sel, 0]
        spherical = np.asarray(diag_out["diag_spherical_charges_raw"])[node_sel]
        energy_short = np.asarray(diag_out["diag_energy_short_raw"])[node_sel]
        energy_long = np.asarray(diag_out["diag_energy_long_raw"])[node_sel]
        positions_sel = positions[node_sel]
        species_sel = species[node_sel]

        max_degree_lr = int(getattr(model, "max_degree_lr", 0))
        by_l = {}
        start = 0
        for l in range(max_degree_lr + 1):
            width = 2 * l + 1
            by_l[f"l{l}"] = spherical[:, start : start + width]
            start += width

        return {
            "graph_id": graph_id,
            "positions": positions_sel,
            "species": species_sel,
            "scalar_charges": scalar,
            "spherical_charges": spherical,
            "spherical_by_l": by_l,
            "energy_short": energy_short,
            "energy_long": energy_long,
        }

    # ... manager preamble

    def get_lr(opt_state):
        return float(opt_state.hyperparams["learning_rate"])

    def report_on_lr(opt_state):
        lr = get_lr(opt_state)
        return f"LR: {lr:.3e}"

    def format_metrics(metrics, keys=["energy", "forces"]):
        key_to_unit = {"energy": "meV/atom", "forces": "meV/Å", "stress": "meV"}
        key_to_name = {"energy": "E", "forces": "F", "stress": "σ"}
        msg = []

        for key in keys:
            m = metrics[key]

            msg.append(f". {key_to_name[key]}")
            if "r2" in m:
                msg.append(f".. R2  : {m['r2']:.3f} %")
            msg.append(f".. MAE : {m['mae']:.3e} {key_to_unit[key]}")
            msg.append(f".. RMSE: {m['rmse']:.3e} {key_to_unit[key]}")

        return msg

    class Manager:
        def __init__(self, state, interval, loggers, workdir, model, baseline, max_steps):
            self.state = state
            self.interval = interval
            self.loggers = loggers
            self.workdir = workdir
            self.model = model
            self.baseline = baseline

            self.max_steps = max_steps

            self.start_step = state["step"]
            self.start_time = monotonic()

            self.cancel = False

        @property
        def done(self):
            return self.step >= self.max_steps or self.cancel

        @property
        def step(self):
            return self.state["step"]

        @property
        def elapsed(self):
            return monotonic() - self.start_time

        @property
        def time_per_step(self):
            return self.elapsed / (self.step - self.start_step)

        @property
        def compute_time_per_step(self):
            return self.time_per_step - pipeline_speed

        @property
        def eta(self):
            return (self.max_steps - self.step) * self.time_per_step

        def should_validate(self, step):
            return step >= self.step + self.interval

        def report(
            self,
            step,
            params,
            opt_state,
            train_state,
            train_loss,
            train_metrics,
            valid_loss,
            valid_metrics,
            info={},
            detailed_info=None,
        ):
            assert step > self.step  # always forward

            self.state["step"] = step
            self.state["opt_state"] = opt_state
            self.state["iter_train"] = train_state

            if jnp.isnan(train_loss):
                comms.warn(f"loss became NaN at step={self.step}, canceling training")
                self.cancel = True

            if get_lr(opt_state) < min_learning_rate:
                # sometimes we stop decay before max_steps, in that case don't break
                if stop_decay_after == max_steps:
                    comms.talk(
                        f"learning rate has reached minimum at step={self.step}, canceling"
                    )
                    self.cancel = True

            info = {
                "lr": get_lr(opt_state),
                "time_per_step": self.time_per_step,
                "compute_time_per_step": self.compute_time_per_step,
                **info,
            }

            for logger in self.loggers:
                logger(
                    self.state["step"],
                    train_loss,
                    train_metrics,
                    valid_loss,
                    valid_metrics,
                    other=info,
                )

            metrics = {"train": train_metrics, "valid": valid_metrics}
            metrics = jax.tree_util.tree_map(lambda x: np.array(x), metrics)

            save_checkpoints(
                metrics,
                params,
                self.state,
                self.model,
                self.baseline,
                self.workdir,
                config=config,
            )

            title = f"state at step: {self.step}"
            msg = []

            msg.append(f"train loss: {train_loss:.5e}")
            msg.append(f"valid loss: {valid_loss:.5e}")

            msg.append(report_on_lr(opt_state))

            msg.append("validation errors:")
            msg += format_metrics(metrics["valid"], keys=keys)

            msg.append("")
            msg.append(f"elapsed: {s_to_string(self.elapsed, 's')}")
            msg.append(
                f"timing: {s_to_string(self.time_per_step)}/step, {s_to_string(self.eta, 'm')} ETA"
            )
            if "diag_scalar_charge_mean" in info:
                msg.append("")
                msg.append("diagnostics (validation):")
                msg.append(
                    f". scalar charge mean      : {info['diag_scalar_charge_mean']:.3e}"
                )
                msg.append(
                    f". scalar charge |.| mean  : {info['diag_scalar_charge_abs_mean']:.3e}"
                )
                msg.append(
                    f". spherical charge |.| mean: {info['diag_spherical_charge_abs_mean']:.3e}"
                )
                msg.append(
                    f". short-range energy mean : {info['diag_energy_short_mean']:.3e}"
                )
                msg.append(
                    f". long-range energy mean  : {info['diag_energy_long_mean']:.3e}"
                )
                msg.append(
                    f". total energy mean       : {info['diag_energy_total_mean']:.3e}"
                )
                if "diag_force_long_over_total" in info:
                    msg.append(
                        f". ||F_long||/||F_total||  : {info['diag_force_long_over_total']:.3e}"
                    )
            if detailed_info is not None:
                diagnostics_dir = self.workdir / "diagnostics"
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
                csv_file = diagnostics_dir / f"step_{self.step:08d}_equivariant_charges.csv"
                txt_file = diagnostics_dir / f"step_{self.step:08d}_summary.txt"

                max_degree_lr = int(getattr(model, "max_degree_lr", 0))
                spherical_cols = [f"sph_{i}" for i in range((max_degree_lr + 1) ** 2)]
                header_cols = [
                    "atom_index",
                    "Z",
                    "x",
                    "y",
                    "z",
                    "scalar_charge",
                    *spherical_cols,
                    "energy_short",
                    "energy_long",
                ]
                rows = []
                for idx in range(detailed_info["scalar_charges"].shape[0]):
                    row = [
                        idx,
                        int(detailed_info["species"][idx]),
                        float(detailed_info["positions"][idx, 0]),
                        float(detailed_info["positions"][idx, 1]),
                        float(detailed_info["positions"][idx, 2]),
                        float(detailed_info["scalar_charges"][idx]),
                        *[
                            float(v)
                            for v in detailed_info["spherical_charges"][idx].tolist()
                        ],
                        float(detailed_info["energy_short"][idx]),
                        float(detailed_info["energy_long"][idx]),
                    ]
                    rows.append(row)
                np.savetxt(
                    csv_file,
                    np.asarray(rows, dtype=float),
                    delimiter=",",
                    header=",".join(header_cols),
                    comments="",
                    fmt="%.10e",
                )

                summary_lines = []
                summary_lines.append(
                    f"step={self.step}, graph_id={detailed_info['graph_id']}, num_atoms={detailed_info['scalar_charges'].shape[0]}"
                )
                summary_lines.append(
                    "scalar_charges: "
                    + np.array2string(
                        detailed_info["scalar_charges"],
                        precision=6,
                        separator=", ",
                        max_line_width=200,
                    )
                )
                for l_name, values in detailed_info["spherical_by_l"].items():
                    summary_lines.append(
                        f"spherical_charges_{l_name}: "
                        + np.array2string(
                            values,
                            precision=6,
                            separator=", ",
                            max_line_width=200,
                        )
                    )
                summary_lines.append(
                    "energy_short: "
                    + np.array2string(
                        detailed_info["energy_short"],
                        precision=6,
                        separator=", ",
                        max_line_width=200,
                    )
                )
                summary_lines.append(
                    "energy_long: "
                    + np.array2string(
                        detailed_info["energy_long"],
                        precision=6,
                        separator=", ",
                        max_line_width=200,
                    )
                )
                txt_file.write_text("\n".join(summary_lines), encoding="utf-8")
                msg.append("")
                msg.append(f"detailed diagnostics saved: {csv_file.as_posix()}")
                msg.append(f"detailed diagnostics saved: {txt_file.as_posix()}")
                msg.append(
                    f"equivariant charges/energies (first valid graph id={detailed_info['graph_id']}):"
                )
                msg.append(
                    ". scalar charges: "
                    + np.array2string(
                        detailed_info["scalar_charges"],
                        precision=6,
                        separator=", ",
                        max_line_width=200,
                    )
                )
                for l_name, values in detailed_info["spherical_by_l"].items():
                    msg.append(
                        f". spherical charges {l_name}: "
                        + np.array2string(
                            values,
                            precision=6,
                            separator=", ",
                            max_line_width=200,
                        )
                    )
                msg.append(
                    ". short-range energies: "
                    + np.array2string(
                        detailed_info["energy_short"],
                        precision=6,
                        separator=", ",
                        max_line_width=200,
                    )
                )
                msg.append(
                    ". long-range energies: "
                    + np.array2string(
                        detailed_info["energy_long"],
                        precision=6,
                        separator=", ",
                        max_line_width=200,
                    )
                )

            msg.append("")
            comms.state(msg, title=title)

    manager = Manager(state, valid_every, loggers, workdir, model, baseline, max_steps)

    @jax.jit
    def do_batch(carry, batch):
        params, opt_state = carry

        loss_and_aux, grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
            params, batch
        )
        loss, aux = loss_and_aux
        updates, opt_state = optimizer.update(grads, opt_state, params, value=loss)

        params = optax.apply_updates(params, updates)

        return (params, opt_state), (loss, aux)

    aggregate_loss = np.mean
    aggregate_aux = tree_stack

    # -- train! --
    import itertools

    reporter.step("🚄", spin=False)

    start = monotonic()

    iter_train_with_prefetch = prefetch_to_device(iter_train, 2)
    iter_valid_with_prefetch = prefetch_to_device(itertools.cycle(data_valid), 2)

    ran_steps = 0
    train_aux = []
    train_loss = []
    report = None
    cache_size = 1
    while True:
        try:
            batch = next(iter_train_with_prefetch)
        except StopIteration:
            comms.talk("exhausted training iterator")
            # break
            manager.cancel = True

        if not manager.done:
            ran_steps += batch.graph_mask.sum()
            (params, opt_state), (loss, aux) = do_batch((params, opt_state), batch)
            current_step = manager.step + ran_steps
            del batch
            reporter.tick(f"{current_step}")

            if do_batch._cache_size() > cache_size:
                cache_size = do_batch._cache_size()
                comms.talk(f"recompiled at step={current_step} ({cache_size})")

            train_aux.append(aux)
            train_loss.append(loss)

        if report is not None:
            manager.report(*report)
            report = None

        if manager.done:
            break

        if manager.should_validate(manager.step + ran_steps):
            valid_aux = []
            valid_loss = []
            valid_diag = []
            detailed_diag = None
            for i in range(len(data_valid)):
                batch = next(iter_valid_with_prefetch)
                reporter.tick(f"{current_step} (valid {i})")

                # for chunk_size > 1, we use the non-vmapped fn here
                loss, aux = jax.jit(_loss_fn)(params, batch)
                diag_out = diagnostics_fn(params, batch)
                force_ratio = force_ratio_fn(params, batch)
                valid_diag.append(
                    {
                        **{k: diag_out[k] for k in diag_keys},
                        "diag_force_long_over_total": force_ratio,
                    }
                )
                if i == 0:
                    detailed_diag = build_detailed_diag(diag_out, batch)

                valid_aux.append(aux)
                valid_loss.append(loss)

            train_aux = aggregate_aux(train_aux)
            train_metrics = train_metrics_fn(train_aux)

            valid_aux = tree_stack(valid_aux)
            valid_metrics = valid_metrics_fn(valid_aux)

            train_loss = aggregate_loss(train_loss)
            valid_loss = np.mean(valid_loss)
            valid_diag = {
                k: float(np.mean([float(np.asarray(d[k])) for d in valid_diag]))
                for k in (*diag_keys, "diag_force_long_over_total")
            }

            report = (
                manager.step + ran_steps,
                params,
                opt_state,
                iter_train.get_state(),
                train_loss,
                train_metrics,
                valid_loss,
                valid_metrics,
                {
                    "epoch": float((manager.step + ran_steps) / batches_per_epoch),
                    "compiles_do_batch": do_batch._cache_size(),
                    "compiles_loss_fn": loss_fn._cache_size(),
                    **valid_diag,
                },
                detailed_diag,
            )

            train_aux = []
            train_loss = []
            ran_steps = 0

    # -- wrap up --
    from marathon.emit import get_all, plot

    reporter.step("wrapup")

    pred_fn = jax.jit(pred_fn)

    def get_batcher():
        return ToFixedShapeBatch(num_graphs=2, num_edges=num_edges, num_nodes=num_nodes)

    test = {}
    for name, (source, save) in test_datasets.items():
        source = DataSource(source, species_to_weight=species_to_weight)
        batcher = get_batcher()

        def it():
            for i in range(len(source)):
                sample = to_sample.map(source[i])
                yield Record(data=sample, metadata=RecordMetadata(index=i, record_key=i))

        batches = [prepare_ewald.map(b.data) for b in batcher(it())]
        test[name] = (batches, save)

    def predict_and_collate(params, batches):
        predictions = {k: [] for k in keys}
        labels = {k: [] for k in keys}

        for batch in batches:
            preds = pred_fn(params, batch)

            for key in keys:
                mask = batch.labels[key + "_mask"]
                if mask.any():
                    predictions[key].append(preds[key][mask])
                    labels[key].append(batch.labels[key][mask])

        final_predictions = {}
        final_labels = {}

        for key in predictions.keys():
            if "energy" in key:
                final_predictions[key] = np.array(predictions[key]).flatten()
            if "forces" in key:
                final_predictions[key] = np.concatenate(predictions[key]).reshape(-1, 3)
            if "stress" in key:
                final_predictions[key] = np.array(predictions[key]).reshape(-1, 3, 3)

        for key in keys:
            if key == "energy":
                final_labels[key] = np.array(labels[key]).flatten()
            if key == "forces":
                final_labels[key] = np.concatenate(labels[key]).reshape(-1, 3)
            if key == "stress":
                final_labels[key] = np.array(labels[key]).reshape(-1, 3, 3)

        return final_labels, final_predictions

    for f, items in get_all(workdir, state):
        if f.suffix == ".backup":
            continue

        comms.talk(f"working on {f}")

        params, _, _, _, metrics, _ = items

        for name, (batches, save) in test.items():
            labels, predictions = predict_and_collate(params, batches)

            out = f / f"plot/{name}"
            out.mkdir(parents=True, exist_ok=True)

            plot(out, predictions, labels)

            if save:
                np.savez_compressed(out / "energy.npz", predictions["energy"])
                np.savez_compressed(out / "forces.npz", predictions["forces"])

    reporter.done()
    if use_wandb:
        run.finish()

    comms.talk("cleaning up")
    import shutil

    if use_wandb:
        wandb_dir = Path("wandb")
        if wandb_dir.is_dir():
            shutil.rmtree(wandb_dir)

    for f, items in get_all(workdir, state):
        if f.suffix == ".backup":
            shutil.rmtree(f)

    comms.state("done!")
