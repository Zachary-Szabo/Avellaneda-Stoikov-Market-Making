import matplotlib.pyplot as plt

ks = [0.0, 0.3, 0.5, 0.7, 1.0]
train_v1  = [-1.686, -0.151, 0.255, 0.368, 0.226]
test_v1   = [-1.369, -0.002, 0.354, 0.439, 0.303]
train_v10 = [-1.684, -0.226, 0.196, 0.321, 0.187]
test_v10  = [-1.405, -0.068, 0.297, 0.394, 0.265]
train_v50 = [-1.689, -0.367, 0.072, 0.221, 0.104]
test_v50  = [-1.448, -0.216, 0.182, 0.306, 0.199]

# Plot 1: Sharpe vs k (Train & Test at min_vol=1 and 50)
plt.figure(figsize=(7, 4.2))
plt.plot(ks, train_v1,  "o-",  label="Train (min_vol=1)")
plt.plot(ks, test_v1,   "o--", label="Test (min_vol=1)")
plt.plot(ks, train_v50, "s-",  label="Train (min_vol=50)")
plt.plot(ks, test_v50,  "s--", label="Test (min_vol=50)")
plt.axhline(0, color="gray", lw=0.8, ls=":")
plt.xlabel("Model weight k  (center = m + k*(fair - m))")
plt.ylabel("Per-window Sharpe")
plt.title("Real-Trade Replay: Sharpe vs. Fair-Value Blend (gamma=0.1)")
plt.legend()
plt.tight_layout()
plt.savefig("sharpe_vs_k.png", dpi=150)

# Plot 2: Sharpe vs min_vol at best k=0.7
vols = [1, 10, 50]
plt.figure(figsize=(6, 4))
plt.plot(vols, [0.368, 0.321, 0.221], "o-",  label="Train (k=0.7)")
plt.plot(vols, [0.439, 0.394, 0.306], "o--", label="Test (k=0.7)")
plt.xlabel("Queue threshold min_vol (contracts strictly through quote)")
plt.ylabel("Per-window Sharpe")
plt.title("Adverse Selection: Sharpe vs. Queue Strictness (k=0.7)")
plt.legend()
plt.tight_layout()
plt.savefig("sharpe_vs_min_vol.png", dpi=150)
print("Saved sharpe_vs_k.png and sharpe_vs_min_vol.png")