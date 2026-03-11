LARGE_PENALTY_CONSTANT = 100000


class SteinerTree():
    def __init__(self, nodes: list[str], edges: list[tuple[str, str, int]], terminals: list[str]) -> None:
        # nodes is a list of strings, where each string represents a node
        # edges is a list of string tuples, where each string in a tuple represents a node
        # terminals is a list of strings, it must be a subset of nodes 
        self._check_validity(nodes, edges, terminals)
        self.nodes = nodes
        self.edges = edges
        self.terminals = terminals

    @staticmethod
    def _check_validity(nodes: list[str], edges: list[tuple[str, str, int]], terminals: list[str]) -> None:
        # check the type of nodes
        # check the contents of nodes
        if not nodes:
            raise ValueError("nodes can't be empty")
        if not (
            isinstance(nodes, list) and
            all(isinstance(node, str) for node in nodes)
        ):
            raise ValueError("invalid format for: nodes")
        
        
        # check the type and content of edges
        if not edges:
            raise ValueError("edges can't be empty")
        if not (
            isinstance(edges, list) and 
            all((
                isinstance(edge, tuple) and
                len(edge) == 3 and
                isinstance(edge[0], str) and
                isinstance(edge[1], str) and
                isinstance(edge[2], int) and
                edge[0] in nodes and
                edge[1] in nodes and
                edge[2] >= 0
            ) for edge in edges)
        ):
            raise ValueError("invalid format for: edges")
        
        #check the type and content of terminals:
        if not terminals:
            raise ValueError("terminals can't be empty")
        if not (
            isinstance(terminals, list) and
            all((
                isinstance(terminal, str) and
                terminal in nodes
            ) for terminal in terminals)
        ):
            raise ValueError("invalid format for: terminals")
        
    def check_edge(self, node1: str, node2: str) -> int:
        # check if the nodes are even in the graph
        if not (node1 in self.nodes and node2 in self.nodes):
            raise ValueError("these nodes are not in this graph")
        
        #if they are the same node weight is 0
        if node1 == node2:
            return 0
       
        # if they are not the same node try to find an edge between them 
        for edge in self.edges:
            if (edge[0] == node1 and edge[1] == node2) or (edge[0] == node2 and edge[1] == node1):
                return edge[2]
            
        # if there is no edge between them return inf
        return LARGE_PENALTY_CONSTANT



        