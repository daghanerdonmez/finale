from typing import Dict, List, Optional, Tuple


class MinVertexCover:
    def __init__(
        self,
        vertices: List[str],
        edges: List[Tuple[str, str]],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._check_validity(vertices, edges, weights)
        self.vertices = vertices
        self.edges = [tuple(sorted((u, v))) for u, v in edges]
        self.weights = {
            vertex: float(weights[vertex]) if weights and vertex in weights else 1.0
            for vertex in vertices
        }

    @staticmethod
    def _check_validity(
        vertices: List[str],
        edges: List[Tuple[str, str]],
        weights: Optional[Dict[str, float]],
    ) -> None:
        if not vertices:
            raise ValueError("vertices can't be empty")
        if not (
            isinstance(vertices, list)
            and all(isinstance(vertex, str) for vertex in vertices)
        ):
            raise ValueError("invalid format for: vertices")
        if len(vertices) != len(set(vertices)):
            raise ValueError("vertices cannot have duplicate names")

        if not edges:
            raise ValueError("edges can't be empty")
        if not (
            isinstance(edges, list)
            and all(
                isinstance(edge, tuple)
                and len(edge) == 2
                and isinstance(edge[0], str)
                and isinstance(edge[1], str)
                and edge[0] in vertices
                and edge[1] in vertices
                for edge in edges
            )
        ):
            raise ValueError("invalid format for: edges")

        seen_edges = set()
        for u, v in edges:
            if u == v:
                raise ValueError("self-loops are not supported")
            key = tuple(sorted((u, v)))
            if key in seen_edges:
                raise ValueError("there cannot be duplicate edges")
            seen_edges.add(key)

        if weights is None:
            return

        if not isinstance(weights, dict):
            raise ValueError("invalid format for: weights")

        unknown_vertices = set(weights) - set(vertices)
        if unknown_vertices:
            raise ValueError("weights contain vertices that are not in the graph")

        if not all(isinstance(weight, (int, float)) and weight >= 0 for weight in weights.values()):
            raise ValueError("weights must be non-negative numbers")

    @staticmethod
    def variable_label(vertex: str) -> str:
        return f"x::{vertex}"

    def selected_vertices_from_sample(self, sample: dict) -> List[str]:
        selected_vertices = []
        for vertex in self.vertices:
            if sample.get(self.variable_label(vertex), 0) == 1:
                selected_vertices.append(vertex)
        return selected_vertices

    def cover_weight(self, selected_vertices: List[str]) -> float:
        return sum(self.weights[vertex] for vertex in selected_vertices)

    def uncovered_edges(self, selected_vertices: List[str]) -> List[Tuple[str, str]]:
        selected_set = set(selected_vertices)
        uncovered = []
        for u, v in self.edges:
            if u not in selected_set and v not in selected_set:
                uncovered.append((u, v))
        return uncovered

    def is_valid_cover(self, selected_vertices: List[str]) -> bool:
        return not self.uncovered_edges(selected_vertices)
