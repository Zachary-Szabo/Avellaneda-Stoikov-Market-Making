"""Reproduce the paper's comparison and the gamma sweep. Usage: python run_experiments.py"""
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from avellaneda_stoikov.params import Params
from avellaneda_stoikov.simulate import run_many
from avellaneda_stoikov.strategies import InventoryStrategy, SymmetricStrategy

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
N = 1000
p = Params()

results = {s.name: run_many(s, p, N, seed=7) for s in (InventoryStrategy(), SymmetricStrategy())}

print(f"{'strategy':<10} {'spread':>7} {'mean P&L':>9} {'std P&L':>8} {'mean q':>7} {'std q':>6}")
for name, rs in results.items():
    pnl = np.array([r.pnl for r in rs]); q = np.array([r.final_q for r in rs])
    spr = np.mean([r.avg_spread for r in rs])
    print(f"{name:<10} {spr:7.2f} {pnl.mean():9.2f} {pnl.std():8.2f} {q.mean():7.2f} {q.std():6.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, rs in results.items():
    axes[0].hist([r.pnl for r in rs], bins=50, alpha=0.6, label=name)
    axes[1].hist([r.final_q for r in rs], bins=range(-30, 31), alpha=0.6, label=name)
axes[0].set_title("Final P&L"); axes[1].set_title("Final inventory")
for ax in axes: ax.legend()
fig.tight_layout(); fig.savefig(OUT / "pnl_and_inventory.png", dpi=120)

gammas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
means, stds = [], []
for g in gammas:
    rs = run_many(InventoryStrategy(), replace(p, gamma=g), 300, seed=11)
    pnl = np.array([r.pnl for r in rs]); means.append(pnl.mean()); stds.append(pnl.std())
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(stds, means, marker="o")
for g, x, y in zip(gammas, stds, means): ax.annotate(f"γ={g}", (x, y), textcoords="offset points", xytext=(5, 5))
ax.set_xlabel("Std of P&L (risk)"); ax.set_ylabel("Mean P&L"); ax.set_title("Risk vs. return as γ changes")
fig.tight_layout(); fig.savefig(OUT / "gamma_sweep.png", dpi=120)
print("gamma sweep:", [(g, round(m, 1), round(s, 1)) for g, m, s in zip(gammas, means, stds)])
