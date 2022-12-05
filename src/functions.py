import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.approximation.vertex_cover import min_weighted_vertex_cover
from qibo.symbols import X, Y, Z, I
from qibo import hamiltonians


def read_parse_graph(graph_file):
    graph = nx.read_graphml(graph_file)
    # ensure nodes are ints starting at 0
    graph = nx.convert_node_labels_to_integers(graph,first_label=0)
    return graph

def get_bits(state, nqubits):
    # given a vector, get most probable state as bitstring
    state_bits = (np.abs(state) ** 2).argmax()
    state_bits = "{0:0{bits}b}".format(state_bits, bits=nqubits)

    return state_bits

def get_max_prob(state):
    # given a vector, get coefficient of most probable state
    max_prob = (np.abs(state) ** 2).max()
    return max_prob

def min_cover_cost_h(graph):
    # as defined in https://arxiv.org/abs/1302.5843
    symbolic_ham = 2*sum([Z(e1)*Z(e2) - Z(e1) - Z(e2) for e1, e2 in graph.edges])
    symbolic_ham += 1*sum([Z(node) for node in graph.nodes])
    ham = hamiltonians.SymbolicHamiltonian(symbolic_ham)
    return ham


def ising_0_ham(graph):
    # standard starting h for adiabatic evo
    symbolic_ham = sum([X(node) for node in graph.nodes])
    ham = hamiltonians.SymbolicHamiltonian(symbolic_ham)

    return ham

def bit_flip_mixer_h(graph, backend=None):
    # the bit flip mixer is also used for the max independent set problem and other problems
    # it is described in https://arxiv.org/abs/1709.03489
    symbolic_ham = 0
    nodes = graph.nodes
    # sum iteration
    for i in nodes:
        neighbours = list(graph.neighbors(i))
        degree = len(neighbours)
    # product iteration
        prod = 1
        for k in neighbours:
            k = int(k)
            prod = prod*(I(k) + Z(k))
    # collect sum
        symbolic_ham += 2**(-1*degree) * X(i) * prod
    # def hamiltonian
    ham = hamiltonians.SymbolicHamiltonian(symbolic_ham, backend=backend)
    return ham

def complete_graph_mixer(graph):
    # introduced in https://arxiv.org/abs/1902.00409 for the max k cover problem
    complete_graph = nx.complete_graph(graph)
    symbolic_ham = sum([X(e1)*X(e2) + Y(e1)*Y(e2) for e1, e2 in complete_graph.edges])
    ham = hamiltonians.SymbolicHamiltonian(symbolic_ham)
    return ham

def mincover_classic(graph):
    # Returns a set of nodes whose weight sum is no more than twice the
    # weight sum of the minimum weight vertex cover.
    # This is an approximation algorithm! It is often wrong!
    mincover = min_weighted_vertex_cover(graph)
    bits = ''
    for node in graph.nodes:
        if node in mincover:
            bits += '0'
        else:
            bits += '1'
    return bits


def draw_graph(graph, bitstring):
    # plot graph and color 0,1 according to bitstring
    # graph (nx.graph)  : networkx graph
    # bitstring (str)   : string of bits (0s and 1s)
    bits = np.array([int(bit) for bit in list(bitstring)])
    zeros = np.where(bits==0)[0]
    ones = np.where(bits==1)[0]

    # positions for all nodes
    pos = nx.spring_layout(graph, seed=3113794652)
    # plot nodes
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    nx.draw_networkx_nodes(graph, pos, nodelist=[node for node in zeros], node_color="tab:red")
    nx.draw_networkx_nodes(graph, pos, nodelist=[node for node in ones], node_color="tab:blue")
    # plot edges
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    plt.show()
