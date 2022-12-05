import mincover
import sys
import networkx

import pytest


@pytest.mark.parametrize('method', ['adiabatic', 'qaoa', 'classic'])
@pytest.mark.parametrize('step', ['0.5', '1e-2'])
@pytest.mark.parametrize('time', ['1', '2'])
@pytest.mark.parametrize('depth', ['1', '2'])
@pytest.mark.parametrize('iters', ['10', '20'])
@pytest.mark.parametrize('mixer', ['bit-flip', 'complete-graph'])
def test_mincover(monkeypatch, method, step, time, depth, iters, mixer):
    with monkeypatch.context() as m:
        # dsatur.graphml is the smallest graph for which dsatur fails
        m.setattr(sys, 'argv', ['mincover', 'graphs/dsatur.graphml',
            '-m', 'adiabatic',
            '--dt', step,
            '--T', time,
            '--depth', depth,
            '--iters', iters,
            '--mix', mixer]
        )
        mincover.main()
