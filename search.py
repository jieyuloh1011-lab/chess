import math, chess, torch
from .encoding import board_to_planes, legal_index_map
from .model import ChessNet

VAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900}


# ---------- alpha-beta (fallback when no trained model exists) ----------
def material(b):
    s = sum(v * (len(b.pieces(p, chess.WHITE)) - len(b.pieces(p, chess.BLACK))) for p, v in VAL.items())
    return s if b.turn == chess.WHITE else -s


def negamax(b, d, a, beta):
    if b.is_checkmate(): return -99999 - d
    if b.is_game_over(): return 0
    if d == 0: return material(b)
    for m in sorted(b.legal_moves, key=lambda m: b.is_capture(m), reverse=True):
        b.push(m); s = -negamax(b, d - 1, -beta, -a); b.pop()
        if s >= beta: return beta
        a = max(a, s)
    return a


def alphabeta_move(b, depth=3):
    best, best_s = None, -10**9
    for m in b.legal_moves:
        b.push(m); s = -negamax(b, depth - 1, -10**9, 10**9); b.pop()
        if s > best_s: best, best_s = m, s
    return best


# ---------- MCTS guided by policy + value nets ----------
class Node:
    __slots__ = ("p", "n", "w", "kids")
    def __init__(self, p): self.p, self.n, self.w, self.kids = p, 0, 0.0, {}


class MCTS:
    def __init__(self, model=None, c_puct=1.5, device="cpu"):
        self.model, self.c, self.dev = model, c_puct, device

    @torch.no_grad()
    def _expand(self, node, b):
        """Returns value from the side-to-move's view."""
        if b.is_checkmate(): return -1.0
        if b.is_game_over(claim_draw=True): return 0.0
        imap = legal_index_map(b)
        x = torch.from_numpy(board_to_planes(b)).unsqueeze(0).to(self.dev)
        logits, v = self.model(x)
        pr = torch.softmax(logits[0, list(imap)], 0).tolist()
        for (i, m), p in zip(imap.items(), pr): node.kids[m] = Node(p)
        return float(v)

    def search(self, board, sims=200):
        root = Node(1.0); self._expand(root, board)
        for _ in range(sims):
            b, node, path = board.copy(), root, [root]
            while node.kids:
                sq = math.sqrt(node.n + 1)
                m, node = max(node.kids.items(), key=lambda kv: (-kv[1].w / kv[1].n if kv[1].n else 0)
                              + self.c * kv[1].p * sq / (1 + kv[1].n))
                b.push(m); path.append(node)
            v = self._expand(node, b) if node.n == 0 else 0.0
            for nd in reversed(path):
                nd.n += 1; nd.w += v; v = -v
        return max(root.kids.items(), key=lambda kv: kv[1].n)[0]


def load_model(path, device="cpu"):
    m = ChessNet().to(device)
    m.load_state_dict(torch.load(path, map_location=device)); return m.eval()
