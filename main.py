import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class CakeDivisionSimulator:
    EPS = 1e-9

    def __init__(self):
        self.results   = {p: [] for p in ['player_1', 'player_2', 'player_3']}
        self.deviations = []

    def generate_random_preferences(self) -> Dict[str, Dict[str, float]]:
        return {
            player: {
                'A': random.uniform(0.1, 2.0),
                'B': random.uniform(0.1, 2.0)
            }
            for player in self.results
        }

    def generate_cake(self, T: int) -> List[str]:
        return [random.choice(['A', 'B']) for _ in range(T)]

    def calculate_utility(self, slice: List[str], prefs: Dict[str, float]) -> float:
        return sum(prefs[c] for c in slice)

    def normalize_utilities(self,
                            cake: List[str],
                            raw: Dict[str, Dict[str, float]]
                           ) -> Dict[str, Dict[str, float]]:
        return {
            p: {c: raw[p][c] / self.calculate_utility(cake, raw[p])
                for c in raw[p]}
            for p in raw
        }

    def last_diminisher_algorithm(
        self,
        cake: List[str],
        prefs: Dict[str, Dict[str, float]]
    ) -> Tuple[List[List[str]], List[float], float]:
        players  = ['player_1', 'player_2', 'player_3']
        assigned = {p: [] for p in players}
        start, T = 0, len(cake)
        remaining = players.copy()

        while len(remaining) > 1:
            k     = len(remaining)
            quota = 1.0 / 3.0
            first = remaining[0]
            cum, end = 0.0, start
            while end < T and cum + self.EPS < quota:
                cum += prefs[first][cake[end]]
                end += 1
            last = first
            for p in remaining[1:]:
                val = self.calculate_utility(cake[start:end], prefs[p])
                if val > quota + self.EPS:
                    cum2, idx = 0.0, start
                    while idx < end and cum2 + self.EPS < quota:
                        cum2 += prefs[p][cake[idx]]
                        idx += 1
                    end, last = idx, p
            assigned[last] = cake[start:end]
            remaining.remove(last)
            start = end

        assigned[remaining[0]] = cake[start:]
        utilities = [self.calculate_utility(assigned[p], prefs[p]) for p in players]
        deviation = sum(abs(u - 1/3) for u in utilities) / 3
        return [assigned[p] for p in players], utilities, deviation

    def run_simulation(self, T: int, N: int, verbose: bool = False):
        for i in range(N):
            cake   = self.generate_cake(T)
            prefs  = self.normalize_utilities(cake, self.generate_random_preferences())
            _, utils, dev = self.last_diminisher_algorithm(cake, prefs)
            for idx, p in enumerate(self.results):
                self.results[p].append(utils[idx])
            self.deviations.append(dev)
            if verbose:
                print(f"Iter {i+1}: utils={utils}, dev={dev:.4f}")
        self.print_final_statistics()
        self.plot_results()

    def print_final_statistics(self):
        for idx, p in enumerate(self.results, 1):
            arr = np.array(self.results[p])
            print(f"Player {idx}: mean={arr.mean():.4f}, "
                  f"min={arr.min():.4f}, max={arr.max():.4f}, std={arr.std():.4f}")
        dev = np.array(self.deviations)
        print(f"Deviation mean={dev.mean():.4f}, max={dev.max():.4f}")

    def plot_results(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        iters = range(1, len(self.results['player_1']) + 1)

        ax1.plot(iters, self.results['player_1'], 'o-', label='P1')
        ax1.plot(iters, self.results['player_2'], 's-', label='P2')
        ax1.plot(iters, self.results['player_3'], '^-', label='P3')
        ax1.axhline(1/3, ls='--', c='r', label='Ideal')
        ax1.set(title='Utilities per Iteration', xlabel='Iteration', ylabel='Utility')
        ax1.legend(); ax1.grid(alpha=.3)

        all_utils = sum(self.results.values(), [])
        ax2.hist(all_utils, bins=20, edgecolor='black')
        ax2.axvline(1/3, ls='--', c='r')
        ax2.set(title='Utility Distribution', xlabel='Utility', ylabel='Frequency')
        ax2.grid(alpha=.3)

        ax3.plot(iters, self.deviations, 'o-', c='orange')
        ax3.set(title='Deviation per Iteration', xlabel='Iteration', ylabel='Deviation')
        ax3.grid(alpha=.3)

        data = [self.results[p] for p in self.results]
        ax4.boxplot(data, labels=['P1', 'P2', 'P3'])
        ax4.axhline(1/3, ls='--', c='r')
        ax4.set(title='Boxplot Utilities', ylabel='Utility')
        ax4.grid(alpha=.3)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    sim = CakeDivisionSimulator()
    sim.run_simulation(T=300, N=50)
