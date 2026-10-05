"""Check whether a seeded OpenJij batch actually explores distinct reads."""
import json
from pathlib import Path

import dimod
import numpy as np
import openjij

from .benchmark_repairs import make_instance, big_m


def main():
    bqm = big_m(make_instance("ladder", 8, 0), 3.0)
    bqm.relabel_variables({v: i for i, v in enumerate(bqm.variables)})
    result = {}
    for name, sampler in (("SA", openjij.SASampler()), ("SQA", openjij.SQASampler())):
        params = {"num_sweeps": 300}
        if name == "SQA":
            params["trotter"] = 4
        fixed = sampler.sample(bqm, num_reads=8, seed=0, **params)
        varied = dimod.concatenate([sampler.sample(bqm, num_reads=1, seed=i, **params) for i in range(8)])
        result[name] = dict(fixed_seed_unique=int(len(np.unique(fixed.record.sample, axis=0))),
            varied_seed_unique=int(len(np.unique(varied.record.sample, axis=0))),
            fixed_seed_energies=fixed.record.energy.tolist(),
            varied_seed_energies=varied.record.energy.tolist(),
            reads=8, parameters=params)
    path = Path(__file__).with_name("openjij_seed_diagnostic.json")
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
