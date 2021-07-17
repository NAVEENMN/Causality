import os
import time
import simpy
import numpy as np
import pandas as pd
import pingouin as pg
import networkx as nx
from utils import Utils
from graph import Graph
from graph import CausalGraph
from itertools import permutations
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class Observations(object):
    def __init__(self, columns):
        self.column_names = columns
        self.observations = dict()
        self.observations = {label: [] for label in columns}

    def add_an_observation(self, observation):
        for col_name in observation.keys():
            self.observations[col_name].append(observation[col_name])

    def get_observations(self):
        df = pd.DataFrame(self.observations)
        return df

    def save_observations(self):
        logging.info('Saving Observations')
        df = pd.DataFrame(self.observations)
        df.to_csv(os.path.join(os.getcwd(), 'data', 'observations.csv'))


class State(object):
    def __init__(self, env, obs, graph):
        self.env = env
        self.graph = graph
        self.obs = obs
        self.action = env.process(self.run())

    def run(self):
        logging.info("Running graph simulation")
        while True:
            try:
                yield self.env.process(self.state_a())
                yield self.env.process(self.state_b())
                yield self.env.process(self.state_c())
            except simpy.Interrupt:
                # When interrupted sample graph values
                self.obs.add_an_observation(self.graph.get_values())

    def state_a(self):
        # Initialize graph independent variables aka source nodes.
        logging.info("*** Init: Initializing independent variables")
        src = self.graph.get_source_nodes()
        for node in src:
            self.graph.set_node_value(node, np.random.randn())
        yield self.env.timeout(1)

    def state_b(self):
        # Run a depth first traversal and update node values
        # Value(child_node) = Value(parent_node) * EdgeValue(p, c) + bias(c)
        logging.info("*** Update: Making DFS updates")
        independent_variables = self.graph.get_source_nodes()

        def depth_first_traversal(graph, node):
            v = graph.get_node_value(node)
            for successor in graph.get_successors(node):
                edge_value = graph.get_edge_value(node, successor)
                w, c = edge_value['weight'], edge_value['capacity']
                value = (v * w) + c
                graph.set_node_value(successor, value.item(0))
                depth_first_traversal(graph, successor)
            return

        for variable in independent_variables:
            depth_first_traversal(self.graph, variable)

        yield self.env.timeout(2)

    def state_c(self):
        logging.info("*** Update: Resetting graph")
        self.graph.reset()
        yield self.env.timeout(3)


def sample_graph(env, cg, num_of_samples):
    for _ in range(num_of_samples):
        logging.info("*** Sample: Taking sample")
        yield env.timeout(1)
        cg.action.interrupt()


def save_graph(causal_graph, testing_graph, predicted_graph, step, attr=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    axes[0].set_title('Original Graph')
    causal_graph.draw(axes=axes[0])
    axes[1].set_title(f'{attr[0]} = {attr[1]}')
    Graph.draw_graph(testing_graph, axes=axes[1])
    axes[2].set_title('Predicted Graph')
    Graph.draw_graph(predicted_graph, axes=axes[2])
    # plt.show()
    fig.savefig(os.path.join(os.getcwd(), 'tmp', f'graph_{step}.png'))
    plt.clf()
    plt.close(fig)


def run_pc_algorithm(causal_graph, observations):
    step = 0
    node_names = causal_graph.get_node_names()
    complete_graph = nx.complete_graph(node_names)

    # case a: empty z
    nodes = permutations(node_names, 2)
    for (x, y) in nodes:
        a, b = np.reshape(observations[x].to_numpy(), (-1, 1)), np.reshape(observations[y].to_numpy(), (-1, 1))
        corr = np.corrcoef(a, b, rowvar=False)
        corr = corr[0][1]
        if complete_graph.has_edge(x, y) and (np.abs(corr) < 0.9):
            complete_graph.remove_edge(x, y)

        G = nx.Graph()
        G.add_nodes_from(node_names)
        G.add_edge(x, y)

        step += 1
        save_graph(causal_graph, G, complete_graph, step, attr=[f'Pcorr({x}-{y})', format(corr, '.2f')])

    # case b: z
    nodes = permutations(node_names, 3)
    for (x, y, z) in nodes:
        # print(f'Partial correlation between {(x, y)} and {z}')
        df = pd.DataFrame({'x': observations[x], 'y': observations[y], 'z': observations[z]})
        result = pg.partial_corr(data=df, x='x', y='y', covar='z').round(3)
        p_value, r_value = result['p-val']['pearson'], result['r']['pearson']
        if p_value > 0.08:
            # print(f'{x, y} - {p_value} - {r_value}')
            if complete_graph.has_edge(x, y):
                complete_graph.remove_edge(x, y)

        G = nx.Graph()
        G.add_nodes_from(node_names)
        G.add_edge(x, y)
        G.add_edge(y, z)

        step += 1
        save_graph(causal_graph, G, complete_graph, step, [f'Pcorr({x}-{y}|{z})', format(p_value, '.2f')])


def main():
    # Generate a random causal graph with certain properties
    causal_graph = CausalGraph()
    causal_graph.set_properties(left_mediators_count=1,
                                right_mediators_count=1,
                                forks_count=0,
                                colliders_count=0)
    causal_graph.generate_random_graph()

    # Setup a process to randomly sample values from causal graph
    obs = Observations(columns=causal_graph.get_node_names())

    # Setup simulation environment
    env = simpy.Environment()
    cg = State(env, obs, causal_graph)

    # Randomly sample the graph
    env.process(sample_graph(env, cg, num_of_samples=100))

    # Run simulation
    env.run(until=1000)

    #obs.save_observations()

    # time.sleep(20)
    #print("Running PC")
    #run_pc_algorithm(causal_graph=causal_graph,
    #                 observations=obs.get_observations())

    #Utils.create_gif()
    #Utils.save_pair_plot(obs.get_observations())
    print(f"Plots saved in {os.path.join(os.getcwd(), 'media')}")
    print('Done')


if __name__ == "__main__":
    main()
