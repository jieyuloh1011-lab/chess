import argparse, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from engine.model import ChessNet

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data.npz"); ap.add_argument("--out", default="model.pt")
ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

d = np.load(a.data)
X, P, V = torch.from_numpy(d["x"]), torch.from_numpy(d["p"]).long(), torch.from_numpy(d["v"])
n = len(X); cut = int(n * 0.95); perm = torch.randperm(n); tr, va = perm[:cut], perm[cut:]
net = ChessNet().to(dev); opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)

def batch(idx): return X[idx].float().to(dev), P[idx].to(dev), V[idx].to(dev)

for ep in range(a.epochs):
    net.train(); tr = tr[torch.randperm(len(tr))]
    for i in range(0, len(tr), a.bs):
        x, p, v = batch(tr[i:i + a.bs]); lp, vp = net(x)
        loss = F.cross_entropy(lp, p) + F.mse_loss(vp, v)   # policy CE + value MSE
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        x, p, v = batch(va[:5000]); lp, vp = net(x)
        print(f"epoch {ep+1}: val CE {F.cross_entropy(lp,p):.3f} MSE {F.mse_loss(vp,v):.3f} "
              f"top1 {(lp.argmax(1)==p).float().mean():.3f}")
    torch.save(net.state_dict(), a.out)
