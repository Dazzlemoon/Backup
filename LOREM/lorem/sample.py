import numpy as np

from collections import namedtuple

from marathon.data.sample import to_labels, Sample

Graph = namedtuple(
    "Graph",
    (
        "edges",
        "nodes",
        "centers",
        "others",
        "info",
        "full_edges",
        "full_centers",
        "full_others",
    ),
)


def to_sample(atoms, cutoff, energy=True, forces=True, stress=False):
    graph = to_graph(atoms, cutoff)

    labels = to_labels(
        atoms,
        energy=energy,
        forces=forces,
        stress=stress,
    )

    return Sample(graph, labels)


def to_graph(atoms, cutoff):
    from vesin import ase_neighbor_list as neighbor_list

    if atoms.pbc.all():
        i, j, D, S = neighbor_list(
            "ijDS", atoms, cutoff
        )  # they follow the R_ij = R_j - R_i convention
        Z = atoms.get_atomic_numbers().astype(int)

        sort_idx = np.argsort(i)
        info = {"cell_shifts": S[sort_idx], "cell": atoms.get_cell().array, "pbc": True}

        full_i = None
        full_j = None
        full_D = None

    else:
        assert not atoms.pbc.any()  # can't treat mixed pbc yet

        i, j, D = neighbor_list(
            "ijD", atoms, cutoff
        )  # they follow the R_ij = R_j - R_i convention
        Z = atoms.get_atomic_numbers().astype(int)

        sort_idx = np.argsort(i)
        info = {"pbc": False, "cell_shifts": np.zeros((len(i), 3), dtype=int)}

        N = len(atoms)
        full_i = np.arange(N).repeat(N)
        full_j = np.tile(np.arange(N), N)
        full_D = atoms.get_all_distances(vector=True).reshape(N * N, 3)

    # special case for sn2 dataset: empty neighborlists get forcibly extended,
    # the cutoff function should take care of it
    if len(i) == 0:
        if len(atoms) > 1:
            i = np.array([0, 1])
            j = np.array([1, 0])
            sort_idx = np.array([0, 1])
            d = atoms.get_distance(0, 1, mic=True, vector=True)
            D = np.array([d, -d])

    if len(i) > 0:
        info["max_neighbors"] = np.unique(i, return_counts=True)[1].max()
    else:
        info["max_neighbors"] = 0

    info["positions"] = atoms.get_positions()

    return Graph(
        D[sort_idx],
        Z,
        i[sort_idx],
        j[sort_idx],
        info,
        full_D,
        full_i,
        full_j,
    )
