"""Assignable mod destinations (Mat.Assign1-3), sampled as whole real combinations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from gaussianfreak.dataset import PresetRecord
from gaussianfreak.fields import ASSIGN_COLUMNS
from gaussianfreak.model.copula import FloatArray

ASSIGN_KEYS = sorted(ASSIGN_COLUMNS)
MIN_DESTINATION_OBSERVATIONS = 12
"""Engine presets with the same active-column pattern needed before pooling all engines."""
_ACTIVE_AMOUNT = 0.5

Trio = tuple[float, ...]
Pattern = tuple[bool, ...]


class DestinationSampler:
    """Which destination a mod column targets depends on whether that column carries modulation, and the three
    destinations are chosen together (e.g. 2306/2307/2308). Each sample therefore takes a real training preset's
    destination trio from presets whose assignable columns (Co5-Co7) are active in the same pattern as the
    sampled amounts: from this engine when it has enough such presets, else from all engines.
    """

    def __init__(
        self,
        keys: Sequence[str],
        engine_records: Sequence[PresetRecord],
        pooled_records: Sequence[PresetRecord],
    ) -> None:
        self.slots = [keys.index(k) for k in ASSIGN_KEYS if k in keys]
        self.take = [ASSIGN_KEYS.index(keys[j]) for j in self.slots]
        self.mod_columns = [
            [i for i, k in enumerate(keys) if k.startswith(ASSIGN_COLUMNS[a] + ".")] for a in ASSIGN_KEYS
        ]
        self.engine = self._table(engine_records)
        self.pooled = self._table(pooled_records)
        self.engine_all = [trio for trios in self.engine.values() for trio in trios]

    @staticmethod
    def _table(records: Sequence[PresetRecord]) -> dict[Pattern, list[Trio]]:
        table: dict[Pattern, list[Trio]] = defaultdict(list)
        for r in records:
            if not all(k in r.fields for k in ASSIGN_KEYS):
                continue
            pattern = tuple(
                any(v.value != 0 for k, v in r.fields.items() if k.startswith(ASSIGN_COLUMNS[a] + "."))
                for a in ASSIGN_KEYS
            )
            table[pattern].append(tuple(float(r.fields[a].value) for a in ASSIGN_KEYS))
        return dict(table)

    def fill(self, out: FloatArray, rng: np.random.Generator) -> None:
        """Write a destination trio into each row of sampled values, in place."""
        if not self.slots:
            return
        for row in out:
            pattern = tuple(
                bool(cols) and bool(np.any(np.abs(row[cols]) > _ACTIVE_AMOUNT)) for cols in self.mod_columns
            )
            pool = self.engine.get(pattern, [])
            if len(pool) < MIN_DESTINATION_OBSERVATIONS:
                pool = self.pooled.get(pattern) or self.engine_all
            trio = pool[rng.integers(len(pool))]
            row[self.slots] = [trio[t] for t in self.take]
