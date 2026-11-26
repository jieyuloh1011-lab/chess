import os, chess
from flask import Flask, jsonify, request, render_template
from engine.search import MCTS, alphabeta_move, load_model

app = Flask(__name__)
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pt")
mcts = MCTS(load_model(MODEL_PATH)) if os.path.exists(MODEL_PATH) else None
print("engine:", "neural MCTS" if mcts else "alpha-beta fallback (no model.pt)")


def state(b):
    status = ("checkmate" if b.is_checkmate() else "stalemate" if b.is_stalemate()
              else "draw" if b.is_game_over(claim_draw=True) else "check" if b.is_check() else "ok")
    return {"fen": b.fen(), "turn": "w" if b.turn else "b", "status": status,
            "legal": [m.uci() for m in b.legal_moves]}


def board_from(req):
    try: return chess.Board(req.get("fen") or chess.STARTING_FEN)
    except ValueError: return None


@app.get("/")
def index(): return render_template("index.html")


@app.post("/api/state")
def api_state():
    b = board_from(request.json); return (jsonify(state(b)) if b else ("bad fen", 400))


@app.post("/api/play")  # validation: only legal moves are applied
def api_play():
    j = request.json; b = board_from(j)
    if not b: return "bad fen", 400
    uci = j.get("move", "")
    if len(uci) == 4 and chess.Move.from_uci(uci + "q") in b.legal_moves: uci += "q"
    try: m = chess.Move.from_uci(uci)
    except ValueError: return jsonify(error="malformed move"), 400
    if m not in b.legal_moves: return jsonify(error="illegal move"), 400
    b.push(m); return jsonify(state(b) | {"move": m.uci()})


@app.post("/api/engine")
def api_engine():
    j = request.json; b = board_from(j)
    if not b or b.is_game_over(): return jsonify(error="no move"), 400
    m = mcts.search(b, int(j.get("sims", 200))) if mcts else alphabeta_move(b, 3)
    return jsonify(move=m.uci(), engine="mcts" if mcts else "alphabeta")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
