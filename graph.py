import random
import numpy as np
import networkx as nx

class Graph(object):
    def __init__(self, nx_graph):
        self.graph = nx_graph
        self.node_size = 800
        self.node_color = '#0D0D0D'
        self.font_color = '#D9D9D9'
        self.edge_color = '#262626'

    def add_node_to_graph(self, node):
        self.graph.add_node(node, value=np.random.randn()/10000.0)

    def add_an_edge_to_graph(self, node_a, node_b):
        self.graph.add_edge(node_a, node_b, color=self.edge_color,
                            weight=np.random.normal(0, 1, 1), capacity=np.random.normal(0, 1, 1))

    def get_graph(self):
        return self.graph

    def get_total_nodes(self):
        return self.graph.number_of_nodes()

    def get_node_values(self):
        values = nx.get_node_attributes(self.graph, 'value').values()
        return values

    def get_edge_value(self, node_a, node_b):
        return self.graph.get_edge_data(node_a, node_b)

    def get_successors(self, node):
        return list(self.graph.successors(node))

    def get_node_value(self, node):
        _nodes = self.graph.nodes()
        return _nodes[node]['value']

    def set_node_value(self, node, value):
        attrs = {node: {'value': value}}
        nx.set_node_attributes(self.graph, attrs)

    def get_random_node(self):
        node = None
        if self.graph.number_of_nodes() != 0:
            node = random.sample(self.graph.nodes(), 1)[0]
        return node

    def reset_all_nodes(self):
        # TODO:reset all node values and edge values.
        for node in self.get_nodes():
            self.set_node_value(node, np.random.randn()/1000)

    def get_source_nodes(self):
        _nodes = []
        for node in self.graph.nodes():
            if self.graph.in_degree(node) == 0:
                _nodes.append(node)
        return _nodes

    def get_nodes(self):
        return self.graph.nodes()

    @classmethod
    def draw_graph(cls, _graph, axes):
        nx.draw(_graph, nx.circular_layout(_graph),
                with_labels=True,
                node_size=500,
                ax=axes)

class CausalGraph(Graph):
    def __init__(self):
        super().__init__(nx.DiGraph())
        self.left_mediators_count = 0
        self.right_mediators_count = 0
        self.forks_count = 0
        self.colliders_count = 0

    def __repr__(self):
        return self.get_graph()

    def set_properties(self, left_mediators_count=0,
                       right_mediators_count=0,
                       forks_count=0, colliders_count=0):
        self.left_mediators_count = left_mediators_count
        self.right_mediators_count = right_mediators_count
        self.forks_count = forks_count
        self.colliders_count = colliders_count

    def create_a_node(self):
        _node = f'n_{self.get_total_nodes()}'
        self.add_node_to_graph(_node)
        return _node

    def reset(self):
        self.reset_all_nodes()

    def add_an_element(self, element='right_mediator'):
        _node = self.get_random_node()
        [x, y, z] = [self.create_a_node() for _ in range(3)]
        if element == 'right_mediator':
            # X -> Y -> Z
            self.add_an_edge_to_graph(x, y)
            self.add_an_edge_to_graph(y, z)
        elif element == 'left_mediator':
            # X <- Y <- Z
            self.add_an_edge_to_graph(z, y)
            self.add_an_edge_to_graph(y, x)
        elif element == 'fork':
            # X <- Y -> Z
            self.add_an_edge_to_graph(y, x)
            self.add_an_edge_to_graph(y, z)
        elif element == 'collider':
            # X -> Y <- Z
            self.add_an_edge_to_graph(x, y)
            self.add_an_edge_to_graph(z, y)
        else:
            print('Unsupported element')

        if _node:
            link = random.sample([x, y, z], 1)[0]
            if random.randint(0, 1):
                self.add_an_edge_to_graph(_node, link)
            else:
                self.add_an_edge_to_graph(link, _node)

    def generate_random_graph(self):
        for _ in range(self.left_mediators_count):
            self.add_an_element(element='left_mediator')

        for _ in range(self.right_mediators_count):
            self.add_an_element(element='right_mediator')

        for _ in range(self.forks_count):
            self.add_an_element(element='fork')

        for _ in range(self.colliders_count):
            self.add_an_element(element='collider')

    def get_node_names(self):
        return self.get_nodes()

    def get_values(self):
        values = {}
        for node in self.get_node_names():
            values[node] = self.get_node_value(node)
        return values

    def draw(self, axes):
        colors = nx.get_edge_attributes(self.graph, 'color').values()
        weights = nx.get_edge_attributes(self.graph, 'weight').values()
        nx.draw(self.graph,
                pos=nx.circular_layout(self.graph),
                with_labels=True,
                edge_color=list(colors),
                width=list(weights),
                node_size=500,
                ax=axes)