"""Explicit population-to-model bindings without copying native model objects.

Bindings and model state are caller-owned runtime material. The returned maps
are new containers; aliases deliberately retain the same underlying object.
No dataset discovery, implicit population normalization or model loading occurs.
"""
from collections.abc import Mapping


def assemble_model_slot_bank(models, population_bindings, shared_bindings):
    """Resolve explicit local/shared slots against a uniquely named model bank.

    An empty population map is valid. Every declared population must have at
    least one local slot. Shared and local slot names must be disjoint. Every
    model must be referenced, avoiding silently dropped training outputs.
    Model values are opaque non-None objects; backend validation is separate.
    """
    if not all(isinstance(value, Mapping) for value in (models, population_bindings, shared_bindings)):
        raise ValueError('Explicit model and binding mappings required')
    if any(not isinstance(key, str) or not key for key in models) or any(value is None for value in models.values()):
        raise ValueError('Nonempty model names and nonmissing model objects required')
    if shared_bindings and not population_bindings:
        raise ValueError('Shared bindings require a declared population')
    used = set()
    bank = {}
    for population, local in population_bindings.items():
        if not isinstance(population, str) or not population or not isinstance(local, Mapping) or not local:
            raise ValueError('Named populations with nonempty local bindings required')
        if set(local) & set(shared_bindings):
            raise ValueError('Local and shared slots overlap')
        slots = {}
        for slot, name in [*local.items(), *shared_bindings.items()]:
            if not isinstance(name, str) or name not in models:
                raise ValueError('Binding references a missing model')
            slots[slot] = models[name]
            used.add(name)
        bank[population] = slots
    if used != set(models):
        raise ValueError('Unreferenced model state')
    return bank


def select_model_slots(bank, population):
    """Return an owned slot map; unknown populations raise KeyError explicitly."""
    if not isinstance(bank, Mapping) or not isinstance(population, str) or not population:
        raise ValueError('Explicit bank and named population required')
    selected = bank[population]
    if not isinstance(selected, Mapping):
        raise ValueError('Population slots must be a mapping')
    return dict(selected)
