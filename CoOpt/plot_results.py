from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path('CoOpt/results')
FINAL_DIR = BASE_DIR / 'final_outputs'
SCATTER_PATH = FINAL_DIR / 'scatter_data.json'
REPR_DIR = FINAL_DIR / 'representatives_grids'
PLOTS_DIR = FINAL_DIR / 'figures'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

from matplotlib import font_manager
font_path = FINAL_DIR / 'TimesNewRoman.ttf'
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path.resolve()))
    font_manager._load_fontmanager(try_read_cache=False)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

SCENARIO_COLORS = {
    : '#01665e',  
    : '#8c510a',  
}


def plot_scatter():
    data = json.loads(SCATTER_PATH.read_text())
    fig, ax = plt.subplots(figsize=(6, 5))
    for scenario, entries in data.items():
        dev = [e['dev_ratio'] for e in entries]
        energy = [e['energy_mwh'] for e in entries]
        ax.scatter(dev, energy, s=15, alpha=0.7, label=scenario, color=SCENARIO_COLORS.get(scenario))
    ax.set_xlabel('Developed Area Ratio')
    ax.set_ylabel('Total Cooling+Heating Energy (MWh)')
    ax.set_title('Developed Ratio vs Energy')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'scatter_dev_energy.png', dpi=300)
    plt.close(fig)

    for scenario, entries in data.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        energy = [e['energy_mwh'] for e in entries]
        balance = [e['energy_balance'] for e in entries]
        ax.scatter(energy, balance, s=15, color=SCENARIO_COLORS.get(scenario))
        ax.set_xlabel('Total Cooling+Heating Energy (MWh)')
        ax.set_ylabel('Energy Balance Index')
        ax.set_title(f'{scenario}: Energy vs Balance')
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f'{scenario}_energy_balance.png', dpi=300)
        plt.close(fig)


LAND_CATEGORIES = ['multi_family', 'industrial', 'road', 'green']
CATEGORY_COLORS = {
    : '#d8b365',  
    : '#5ab4ac',    
    : '#8c8c8c',          
    : '#66a61e',         
}


def classify_land(row: dict) -> str:
    values = {
        : row.get('multi_family_ha', 0.0),
        : row.get('facility_industrial_ha', 0.0),
        : row.get('road_area_ha', 0.0),
        : row.get('parks_green_ha', 0.0),
    }
    return max(values, key=values.get)


def plot_representative_grids():
    for path in REPR_DIR.glob('*.json'):
        data = json.loads(path.read_text())
        cells = data['cells']
        rows = sorted({c['row'] for c in cells})
        cols = sorted({c['col'] for c in cells})
        row_index = {r: i for i, r in enumerate(rows)}
        col_index = {c: j for j, c in enumerate(cols)}
        nrows, ncols = len(rows), len(cols)

        eui = np.zeros((nrows, ncols))
        land = np.empty((nrows, ncols), dtype=object)
        for cell in cells:
            i = row_index[cell['row']]
            j = col_index[cell['col']]
            total = 0.0
            for key in ['cooling_kwh_per_m2', 'heating_kwh_per_m2', 'other_electricity_kwh_per_m2']:
                total += cell.get(key, 0.0) or 0.0
            eui[i, j] = total
            land[i, j] = classify_land(cell)

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        im = axes[0].imshow(eui, cmap='viridis', origin='lower')
        axes[0].set_title('EUI (kWh/m²·yr)')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        cmap = [CATEGORY_COLORS[c] for c in LAND_CATEGORIES]
        color_map = {cat: CATEGORY_COLORS[cat] for cat in LAND_CATEGORIES}
        land_rgb = np.zeros((nrows, ncols, 3))
        for cat, color in color_map.items():
            mask = land == cat
            rgb = tuple(int(color.lstrip('#')[k:k+2], 16)/255 for k in (0, 2, 4))
            land_rgb[mask] = rgb
        axes[1].imshow(land_rgb, origin='lower')
        axes[1].set_title('Land-use Dominant Type')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        handles = [plt.Line2D([0], [0], marker='s', color='w', label=cat,
                              markerfacecolor=color_map[cat], markersize=8)
                   for cat in LAND_CATEGORIES]
        axes[1].legend(handles=handles, loc='upper right', fontsize=8, frameon=False)

        fig.suptitle(f"{data['scenario']} - {data['label']} (ind {data['ind_id']})")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(PLOTS_DIR / f"{data['scenario']}_{data['label']}.png", dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    plot_scatter()
    plot_representative_grids()
