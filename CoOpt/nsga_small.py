

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from GeneGenera.decoder import CellGene, GlobalGene
from GeneGenera.evaluator import GeneEvaluator

from CoOpt import config

SEED = int(os.getenv("COOPT_SEED", 123))
RNG = np.random.default_rng(SEED)
POP_SIZE = int(os.getenv("COOPT_POP_SIZE", 20))
N_GEN = int(os.getenv("COOPT_N_GEN", 10))
SCENARIO = os.getenv("COOPT_SCENARIO", "default")
RESULT_DIR = Path("CoOpt/results") / SCENARIO
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def clone_cell(g: CellGene) -> CellGene:
    return CellGene(**g.__dict__)


def mutate_cell(g: CellGene) -> CellGene:
    cell = clone_cell(g)
    if RNG.random() < 0.5:
        cell.coverage_factor = float(np.clip(
            cell.coverage_factor + RNG.normal(0, 0.02),
            *config.CELL_BOUNDS["coverage_factor"],
        ))
    if RNG.random() < 0.5:
        low, high = config.CELL_BOUNDS["floor_base"]
        cell.floor_base = int(np.clip(cell.floor_base + RNG.integers(-1, 2), low, high - 1))
    if RNG.random() < 0.3:
        cell.floor_gradient = float(np.clip(
            cell.floor_gradient + RNG.normal(0, 0.02),
            *config.CELL_BOUNDS["floor_gradient"],
        ))
    if RNG.random() < 0.3:
        cell.podium_ratio = float(np.clip(
            cell.podium_ratio + RNG.normal(0, 0.02),
            *config.CELL_BOUNDS["podium_ratio"],
        ))
    if RNG.random() < 0.4:
        mix = RNG.dirichlet([0.05, 0.4, 0.45, 0.1])[:3]
        cell.mix_seed = tuple(float(x) for x in mix)
    if RNG.random() < 0.3:
        cell.green_ratio = float(np.clip(
            cell.green_ratio + RNG.normal(0, 0.01),
            *config.CELL_BOUNDS["green_ratio"],
        ))
    if RNG.random() < 0.3:
        cell.road_width_factor = float(np.clip(
            cell.road_width_factor + RNG.normal(0, 0.01),
            *config.CELL_BOUNDS["road_width_factor"],
        ))
    if RNG.random() < 0.3:
        cell.road_grid_density = float(np.clip(
            cell.road_grid_density + RNG.normal(0, 0.01),
            *config.CELL_BOUNDS["road_grid_density"],
        ))
    return cell


def mutate_global(g: GlobalGene) -> GlobalGene:
    val = g.__dict__.copy()
    for key, (low, high) in config.GLOBAL_BOUNDS.items():
        if RNG.random() < 0.4:
            sigma = (high - low) * 0.05
            val[key] = float(np.clip(val[key] + RNG.normal(0, sigma), low, high))
    return GlobalGene(**val)


@dataclass
class Individual:
    cells: List[CellGene]
    global_gene: GlobalGene
    ind_id: int
    summary: dict | None = None
    features_stats: dict | None = None
    fitness: tuple[float, float ] | None = None
    feasible: bool = False
    penalty: float = 0.0
    outliers: list[str] | None = None


class SimpleNSGA:
    def __init__(self) -> None:
        self.evaluator = GeneEvaluator()
        self.next_ind_id = 0

    def _make_individual(self, cells: List[CellGene], global_gene: GlobalGene) -> Individual:
        ind = Individual(cells=cells, global_gene=global_gene, ind_id=self.next_ind_id)
        self.next_ind_id += 1
        return ind

    def random_individual(self) -> Individual:
        cells = [config.sample_cell_gene(RNG, gid) for gid in config.GRID_IDS]
        global_gene = config.sample_global_gene(RNG)
        return self._make_individual(cells, global_gene)

    def evaluate(self, ind: Individual) -> None:
        report = self.evaluator.evaluate(ind.cells, ind.global_gene)
        summary = report["summary"]
        ind.summary = summary
        ind.outliers = report["outlier_flags"]
        ind.features_stats = self._summarize_features(report["features"])
        penalty = summary.get("penalty", 0.0)
        ind.penalty = penalty
        if penalty > 0:
            ind.fitness = (1e9, 1e9)
            ind.feasible = False
        else:
            energy = summary["cooling_kwh_per_m2_total_mwh"] + summary["heating_kwh_per_m2_total_mwh"]
            balance = summary["energy_balance_index"]
            ind.fitness = (energy, balance)
            ind.feasible = True

    def crossover(self, p1: Individual, p2: Individual) -> Individual:
        cells = []
        for g1, g2 in zip(p1.cells, p2.cells):
            chosen = g1 if RNG.random() < 0.5 else g2
            cells.append(mutate_cell(chosen))
        global_gene = mutate_global(p1.global_gene if RNG.random() < 0.5 else p2.global_gene)
        return self._make_individual(cells, global_gene)

    def run(self) -> list[Individual]:
        population = [self.random_individual() for _ in range(POP_SIZE)]
        for ind in population:
            self.evaluate(ind)
        for gen in range(N_GEN):
            offspring: list[Individual] = []
            while len(offspring) < POP_SIZE:
                pa, pb = RNG.choice(population, 2, replace=False)
                child = self.crossover(pa, pb)
                self.evaluate(child)
                offspring.append(child)
            population = self.select(population + offspring)
            best = min(population, key=lambda ind: ind.fitness[0])
            print(f"Gen {gen+1}: best cooling+heating={best.fitness[0]:.2f}, penalty={best.penalty:.2f}")
            self.save_frontier(population, gen + 1)
        return population

    def select(self, pool: list[Individual]) -> list[Individual]:
        fronts = self.fast_non_dominated_sort(pool)
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= POP_SIZE:
                new_pop.extend(front)
            else:
                distances = self.crowding_distance(front)
                order = np.argsort(-distances)
                for idx in order:
                    if len(new_pop) >= POP_SIZE:
                        break
                    new_pop.append(front[idx])
                break
        return new_pop

    def fast_non_dominated_sort(self, individuals: list[Individual]) -> list[list[Individual]]:
        fronts: list[list[Individual]] = []
        domination_counts = {id(ind): 0 for ind in individuals}
        dominates = {id(ind): [] for ind in individuals}
        for i in individuals:
            for j in individuals:
                if i is j:
                    continue
                if self._dominates(i, j):
                    dominates[id(i)].append(j)
                elif self._dominates(j, i):
                    domination_counts[id(i)] += 1
            if domination_counts[id(i)] == 0:
                if not fronts:
                    fronts.append([])
                fronts[0].append(i)
        level = 0
        while level < len(fronts):
            next_front: list[Individual] = []
            for ind in fronts[level]:
                for dominated in dominates[id(ind)]:
                    domination_counts[id(dominated)] -= 1
                    if domination_counts[id(dominated)] == 0:
                        next_front.append(dominated)
            if next_front:
                fronts.append(next_front)
            level += 1
        return fronts

    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        f1 = ind1.fitness
        f2 = ind2.fitness
        return all(a <= b for a, b in zip(f1, f2)) and any(a < b for a, b in zip(f1, f2))

    def crowding_distance(self, front: list[Individual]) -> np.ndarray:
        size = len(front)
        if size == 0:
            return np.array([])
        distances = np.zeros(size)
        for m in range(2):
            front_sorted = sorted(front, key=lambda ind: ind.fitness[m])
            distances[front.index(front_sorted[0])] = distances[front.index(front_sorted[-1])] = np.inf
            fmin = front_sorted[0].fitness[m]
            fmax = front_sorted[-1].fitness[m]
            if fmax == fmin:
                continue
            for i in range(1, size - 1):
                prev = front_sorted[i - 1].fitness[m]
                nxt = front_sorted[i + 1].fitness[m]
                distances[front.index(front_sorted[i])] += (nxt - prev) / (fmax - fmin)
        return distances

    def _summarize_features(self, df) -> dict:
        cols = [
            ,
            ,
            ,
            ,
            ,
            ,
            ,
        ]
        stats = {}
        for col in cols:
            if col not in df.columns:
                continue
            stats[f"{col}_mean"] = float(df[col].mean())
            stats[f"{col}_sum"] = float(df[col].sum())
        return stats

    def _serialize_genes(self, ind: Individual) -> dict:
        def serialize_cell(cell: CellGene) -> dict:
            data = asdict(cell)
            if isinstance(data.get("mix_seed"), tuple):
                data["mix_seed"] = list(data["mix_seed"])
            return data

        return {
            : ind.ind_id,
            : [serialize_cell(cell) for cell in ind.cells],
            : asdict(ind.global_gene),
        }

    def save_frontier(self, population: list[Individual], gen: int) -> None:
        front = self.fast_non_dominated_sort(population)[0]
        payload = []
        for ind in front:
            payload.append({
                : ind.ind_id,
                : ind.fitness,
                : ind.summary,
                : ind.penalty,
                : ind.outliers,
                : ind.features_stats,
                : self._serialize_genes(ind),
            })
        out_path = RESULT_DIR / f"frontier_gen{gen:03d}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    ga = SimpleNSGA()
    final_pop = ga.run()
    front = ga.fast_non_dominated_sort(final_pop)[0]
    payload = []
    for ind in front:
        payload.append({
            : ind.ind_id,
            : ind.fitness,
            : ind.summary,
            : ind.penalty,
            : ind.outliers,
            : ind.features_stats,
            : ga._serialize_genes(ind),
        })
    out_path = RESULT_DIR / "final_frontier.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved Pareto front to {out_path}")


if __name__ == "__main__":
    main()
