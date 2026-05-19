from typing import Tuple

from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
from SteinerTreeProblemQUBO.MyFormulization.steiner_to_bqm_daghan import (
    steiner_to_bqm_daghan,
)


def _format_variable_label(variable) -> str:
    if isinstance(variable, tuple):
        return "::".join(str(part) for part in variable)
    return str(variable)


def steiner_to_oj_qubo_daghan(
    problem: SteinerTree,
    constraint_weight: float,
    version = 2
) -> Tuple[dict, float]:
    bqm = steiner_to_bqm_daghan(problem, constraint_weight, version)
    relabeling = {variable: _format_variable_label(variable) for variable in bqm.variables}
    labeled_bqm = bqm.relabel_variables(relabeling, inplace=False)
    return labeled_bqm.to_qubo()
