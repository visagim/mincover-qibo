import numpy as np
import functions
import argparse

from qibo import callbacks, models

class minCover():


    def __init__(self, graph_file, method, backend=None):
        # initialize parameters
        if backend is None:
            from qibo.backends import GlobalBackend
            self.backend = GlobalBackend()
        else:
            self.backend = backend
        self.method = method
        self.graph = functions.read_parse_graph(graph_file)


    def get_hamiltonians(self):
        # objective hamiltonian is the same for both adiabatic, qaoa
        h1 = functions.min_cover_cost_h(self.graph)
        if self.method == "adiabatic":
            h0 = functions.ising_0_ham(self.graph)

        elif self.method == "qaoa":
            h0 = functions.bit_flip_mixer_h(self.graph)
        return h0, h1


    def optimize(self, **params):
        # solve the min vertex cover problem
        print('Running method {}'.format(self.method))
        if self.method == "adiabatic":
            time = params['T']
            step = params['dt']
            final_state, prob = self.run_adiabatic(time, step)

        elif self.method == 'qaoa':
            depth = params['depth']
            iters = params['iters']
            mixer = params['mix']
            final_state, prob = self.run_qaoa(depth, iters, mixer)

        elif self.method == 'classic':
            final_state = functions.mincover_classic(self.graph)
            prob = None
        else:
            raise Exception('Method {} not found. Available methods are classic, adiabatic, and qaoa'.format(self.method))

        print("Found optimal state {} with probability {}".format(final_state,prob))

        if params['plot'] == True:
            # plot result graph
            functions.draw_graph(self.graph, final_state)


    def run_adiabatic(self, time, step):
        # run simulation of adiabatic evolution
        # time (float)  : adiabatic evolution time
        # step (float)  : evolution integration step
        # other parameters are left as default, see qibo/models/evolution.py for more options
        h0, h1 = self.get_hamiltonians()

        nqubits = len(self.graph.nodes)
        evolve = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=step)

        # get initial state and evolve
        initial_state = h0.ground_state()
        final_state = evolve(final_time=time, initial_state=initial_state)
        bit_state = functions.get_bits(final_state, nqubits)
        prob = functions.get_max_prob(final_state)

        return bit_state, prob


    def run_qaoa(self, depth, iters, mixer):
        # qaoa for the min cover problem
        nqubits = len(self.graph.nodes)
        initial_parameters = 0.5 * np.random.random(2*depth)

        hm, hp = self.get_hamiltonians()
        qaoa = models.QAOA(hp, mixer=hm)

        # a proposed initial state is Sum |1>
        initial_state = np.ones(2**nqubits) / np.sqrt(2**nqubits)
        best_energy, final_parameters, extra = qaoa.minimize(initial_parameters,
            method="BFGS",
            options={"maxiter": iters},
        )
        qaoa.set_parameters(final_parameters)
        final_state = qaoa.execute(initial_state)
        bit_state = functions.get_bits(final_state, nqubits)
        prob = functions.get_max_prob(final_state)

        return bit_state, prob


def main():
    parser = argparse.ArgumentParser(description="solve the min vertex cover problem using adiabatic evolution or QAOA")
    parser.add_argument("graph", type=str)
    parser.add_argument("-m", "--method", default="adiabatic", type=str, help="options: adiabatic, QAOA")
    parser.add_argument("--dt", default=1e-1, type=float, help="annealing simulation step")
    parser.add_argument("--T", default=5, type=float, help="annealing time")
    parser.add_argument("--depth", default=4, type=int, help="QAOA depth")
    parser.add_argument("--iters", default=60, type=int, help="QAOA iterations")
    parser.add_argument("--mix", default='complete-graph', type=str, help="QAOA mixer: bit-flip, complete-graph")
    parser.add_argument("--plot", default=False, type=bool, help="Plot resulting graph")
    args = parser.parse_args()
    graph_file = args.graph
    method = args.method
    problem = minCover(graph_file, method)
    problem.optimize(**vars(args))

if __name__ == "__main__":
    main()
