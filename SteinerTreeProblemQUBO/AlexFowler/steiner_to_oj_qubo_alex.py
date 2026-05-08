from typing import Tuple

from SteinerTreeProblemQUBO.MyFormulization import steiner_to_bqm_daghan
from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
from SteinerTreeProblemQUBO.AlexFowler.steiner_to_bqm_alex import (
    steiner_to_bqm_ordering,
)


def _format_variable_label(variable) -> str:
    if isinstance(variable, tuple):
        return "::".join(str(part) for part in variable)
    return str(variable)


def steiner_to_oj_qubo_alex(
    problem: SteinerTree,
    constraint_weight: float,
) -> Tuple[dict, float]:
    bqm = steiner_to_bqm_ordering(problem, constraint_weight)
    relabeling = {variable: _format_variable_label(variable) for variable in bqm.variables}
    labeled_bqm = bqm.relabel_variables(relabeling, inplace=False)
    return labeled_bqm.to_qubo()
