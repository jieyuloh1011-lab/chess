"""PGN (.pgn or .pgn.zst) -> npz of (planes, policy index, value).
Value is stored from the side-to-move's view: +1 mover won, 0 draw, -1 mover lost
(i.e. white result 1/0/-1, sign-flipped on Black's turns)."""
import argparse, io, sys, numpy as np, chess.pgn
sys.path.insert(0, ".")
from engine.encoding import board_to_planes, move_to_index

ap = argparse.ArgumentParser()
ap.add_argument("pgn"); ap.add_argument("--out", default="data.npz")
ap.add_argument("--min_elo", type=int, default=2300)
ap.add_argument("--max_pos", type=int, default=500_000)
a = ap.parse_args()

f = open(a.pgn, "rb")
if a.pgn.endswith(".zst"):
    import zstandard; f = zstandard.ZstdDecompressor().stream_reader(f)
f = io.TextIOWrapper(f, encoding="utf-8", errors="ignore")

X, P, V = [], [], []
res = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
while len(X) < a.max_pos:
    g = chess.pgn.read_game(f)
    if g is None: break
    h = g.headers
    try: ok = min(int(h["WhiteElo"]), int(h["BlackElo"])) >= a.min_elo and h["Result"] in res
    except (KeyError, ValueError): ok = False
    if not ok: continue
    z, b = res[h["Result"]], g.board()
    for ply, m in enumerate(g.mainline_moves()):
        if ply >= 6:  # skip opening-book noise
            X.append(board_to_planes(b).astype(np.float16))
            P.append(move_to_index(m, b.turn)); V.append(z if b.turn == chess.WHITE else -z)
        b.push(m)
np.savez_compressed(a.out, x=np.array(X), p=np.array(P, np.int16), v=np.array(V, np.float32))
print("positions:", len(X))
