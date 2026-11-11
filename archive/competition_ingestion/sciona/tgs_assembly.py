"""Independent reconstruction of TGS boundary-neighbor mosaic assembly.

Source method: Generate_Mosaic.R, revision
2f81d4dd8d50a01579e5f7650259dde92c5c3b8d. Preserves the source's join
multiplicities (it is not a reciprocal-nearest-neighbor test). Corrects zero
index export and rejects contradictory tile placements instead of overwriting.
"""
import numpy as np
from scipy.spatial import cKDTree

from sciona.tgs_boundaries import boundary_features


def _neighbors(reference, query):
    distances, indices = cKDTree(reference).query(query, k=2, workers=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        compatibility = 1 - distances[:, 0] / distances[:, 1]
    return indices[:, 0], distances[:, 0], compatibility


def candidate_edges(reference, query, distance_limit=10., compatibility_limit=0.25):
    """Preserve the source join: forward row i joins all reverse winners of i.

    Duplicate pairs remain significant: the source strip traversal only follows
    a direction when exactly one joined row matches. Equal nearest distances
    have zero compatibility; all-zero distances have undefined compatibility.
    Both are rejected by the strict threshold without arbitrary tie selection.
    """
    a, b = np.asarray(reference, dtype=float), np.asarray(query, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or min(a.shape) < 2:
        raise ValueError('boundary matrices must align and have at least two rows/columns')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('boundary matrices must be finite')
    if not np.isfinite(distance_limit) or distance_limit <= 0:
        raise ValueError('distance limit must be finite and positive')
    if not np.isfinite(compatibility_limit) or not 0 <= compatibility_limit < 1:
        raise ValueError('compatibility limit must be in [0,1)')
    forward, distances, confidence = _neighbors(a, b)
    reverse, _, reverse_confidence = _neighbors(b, a)
    multiplicity = np.bincount(reverse[np.isfinite(reverse_confidence) &
                                       (reverse_confidence > compatibility_limit)], minlength=len(a))
    edges = []
    for origin, target in enumerate(forward):
        if origin != target and distances[origin] < distance_limit and confidence[origin] > compatibility_limit:
            edges.extend([(origin, int(target))] * int(multiplicity[origin]))
    return edges


def _strips(edges, population):
    visited = set()
    strips = []
    for seed in range(population):
        if seed in visited:
            continue
        chain = [seed]
        for reverse in (False, True):
            current = seed
            while True:
                choices = [a if reverse else b for a, b in edges
                           if (b if reverse else a) == current]
                if len(choices) != 1 or choices[0] in chain:
                    break
                current = choices[0]
                if reverse:
                    chain.append(current)
                else:
                    chain.insert(0, current)
                visited.add(current)
        if len(chain) > 1:
            strips.append(chain)
    return strips


def assemble_edges(population, x_edges, y_edges):
    """Construct row/column grids, -1 for holes, from source-axis edge lists."""
    if isinstance(population, bool) or not isinstance(population, (int, np.integer)) or population < 2:
        raise ValueError('population must be an integer of at least two')
    for edges in (x_edges, y_edges):
        for a, b in edges:
            if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= v < population for v in (a, b)) or a == b:
                raise ValueError('invalid mosaic edge')
    strips = [_strips(x_edges, population), [list(reversed(s)) for s in _strips(y_edges, population)]]
    membership = [{tile: strip for strip in axis for tile in strip} for axis in strips]
    adjacency = {i: set() for i in range(population)}
    for a, b in [*x_edges, *y_edges]:
        adjacency[a].add(b)
        adjacency[b].add(a)
    components = []
    unseen = set(adjacency)
    while unseen:
        seed = min(unseen)
        component, pending = set(), [seed]
        while pending:
            tile = pending.pop()
            if tile in component:
                continue
            component.add(tile)
            pending.extend(adjacency[tile] - component)
        unseen -= component
        components.append(component)
    grids = []
    for component in sorted(components, key=lambda c: (-len(c), min(c))):
        if len(component) == 1:
            continue
        positions = {min(component): (0, 0)}
        pending = [min(component)]
        for tile in pending:
            for axis in range(2):
                strip = membership[axis].get(tile, [])
                for offset, neighbor in enumerate(strip):
                    proposed = list(positions[tile])
                    proposed[axis] += offset - strip.index(tile)
                    proposed = tuple(proposed)
                    if neighbor in positions and positions[neighbor] != proposed:
                        raise ValueError('inconsistent mosaic cycle')
                    if neighbor not in positions:
                        positions[neighbor] = proposed
                        pending.append(neighbor)
        if set(positions) != component:
            raise ValueError('ambiguous source strips do not place complete component')
        for edges, expected in ((x_edges, (1, 0)), (y_edges, (0, -1))):
            for a, b in edges:
                if a in component and tuple(np.subtract(positions[a], positions[b])) != expected:
                    raise ValueError('mosaic placement contradicts candidate edge')
        if len(set(positions.values())) != len(positions):
            raise ValueError('mosaic tiles collide')
        coordinates = np.array(list(positions.values()))
        coordinates -= coordinates.min(axis=0)
        # Unlike the R export, allocate max+1 and retain coordinate zero.
        width, height = coordinates.max(axis=0) + 1
        grid = np.full((height, width), -1, dtype=np.int64)
        for tile, (x, y) in zip(positions, coordinates):
            grid[y, x] = tile
        grids.append(grid)
    return grids


def construct_mosaics(images):
    """Build mosaics from finite population/x/y arrays; no file identities."""
    features = boundary_features(images)
    x_edges = candidate_edges(features['d'], features['u'])
    y_edges = candidate_edges(features['l'], features['r'])
    return assemble_edges(len(features['u']), x_edges, y_edges)
