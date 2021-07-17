import os
import numpy as np
import pandas as pd
from utils import Utils
from graph import CausalGraph
from pc_algorithm import PC
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
        logging.info("*** Saving: observations")
        df = pd.DataFrame(self.observations)
        df.to_csv(os.path.join(os.getcwd(), 'data', 'observations.csv'))


class CGSimulation(object):
    def __init__(self, obs, graph, noise=(0.0, 1.0), noise_scale=0.8):
        self.graph = graph
        self.obs = obs
        self.noise_mu = noise[0]
        self.noise_sigma = noise[1]
        self.noise_scale = noise_scale

    def run(self, count):

        def depth_first_traversal(graph, node):
            v = graph.get_node_value(node)
            for successor in graph.get_successors(node):
                edge_value = graph.get_edge_value(node, successor)
                w, c = edge_value['weight'], edge_value['capacity']
                value = (v * w) + c
                _noise = np.random.normal(self.noise_mu, self.noise_sigma, 1)
                value += (_noise*self.noise_scale)
                graph.set_node_value(successor, value.item(0))
                depth_first_traversal(graph, successor)
            return

        for _ in range(count):
            # Initialize graph independent variables aka source nodes.
            logging.info("*** Init: Initializing independent variables")
            src = self.graph.get_source_nodes()
            for node in src:
                mu = np.random.normal(0.5, 1, 1)
                _v = np.random.normal(mu, 1, 1)
                self.graph.set_node_value(node, _v.item(0))

            # Run a depth first traversal and update node values
            # Value(child_node) = Value(parent_node) * EdgeValue(p, c) + bias(c)
            logging.info("*** Update: Making DFS updates")
            independent_variables = self.graph.get_source_nodes()
            for variable in independent_variables:
                depth_first_traversal(self.graph, variable)

            # Collect and save observations
            self.obs.add_an_observation(self.graph.get_values())
            self.obs.save_observations()

            # Reset graph
            logging.info("*** Update: Resetting graph")
            self.graph.reset()


def main():
    # Generate a random causal graph with certain properties
    causal_graph = CausalGraph()
    causal_graph.set_properties(left_mediators_count=1,
                                right_mediators_count=0,
                                forks_count=1,
                                colliders_count=1)
    causal_graph.generate_random_graph()
    causal_graph.save()

    # Setup simulation environment and record observations
    obs = Observations(columns=causal_graph.get_node_names())
    cg = CGSimulation(obs, causal_graph)
    cg.run(1000)

    # Run PC algorithm
    pc = PC(p_value=0.9,
            graph=causal_graph,
            observations=obs.get_observations())
    pc.run()

    Utils.create_gif(loc='pc_0_9')
    Utils.save_pair_plot(obs.get_observations())
    print(f"Plots saved in {os.path.join(os.getcwd(), 'media')}")
    print('Done')


if __name__ == "__main__":
    main()
