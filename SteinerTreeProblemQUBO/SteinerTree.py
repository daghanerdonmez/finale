class SteinerTree():
    def __init__(self, nodes: list[str], edges: list[tuple[str,str, int]], terminals: list[str]) -> None:
        # nodes is a list of strings, where each string represents a node
        # edges is a list of string tuples, where each string in a tuple represents a node
        # terminals is a list of strings, it must be a subset of nodes 
        self._check_validity(nodes, edges, terminals)
        self.nodes = nodes
        self.edges = edges
        self.terminals = terminals

    @staticmethod
    def _check_validity(nodes: list[str], edges: list[tuple[str,str, int]], terminals: list[str]) -> None:
        # check the type of nodes
        # check the contents of nodes
        if not (
            isinstance(nodes, list) and
            all(isinstance(node, str) for node in nodes)
        ):
            raise ValueError("invalid format for: nodes")
        
        
        # check the type and content of edges
        if not (
            isinstance(edges, list) and 
            all((
                isinstance(edge, tuple) and
                len(edge) == 3 and
                isinstance(edge[0], str) and
                isinstance(edge[1], str) and
                isinstance(edge[2], int) and
                edge[0] in nodes and
                edge[1] in nodes 
            ) for edge in edges)
        ):
            raise ValueError("invalid format for: edges")
        
        #check the type and content of terminals:
        if not (
            isinstance(terminals, list) and
            all((
                isinstance(terminal, str) and
                terminal in nodes
            ) for terminal in terminals)
        ):
            raise ValueError("invalid format for: terminals")


        