# Chess Engine (python-chess + ResNet policy/value net + MCTS)

Browser-playable chess engine. Rules, legality and validation come from `python-chess`;
a supervised ResNet learns from master games; MCTS (PUCT) uses it at play time.
With no `model.pt` the app falls back to a depth-3 alpha-beta search, so it always works.

## Layout
| Path | Purpose |
|---|---|
| `engine/encoding.py` | board → 18×8×8 planes (side-to-move perspective), move → 0..4095 index |
| `engine/model.py` | ResNet backbone, policy head (CE) + value head (tanh, MSE) |
| `engine/search.py` | MCTS with the nets + alpha-beta fallback |
| `scripts/prepare_data.py` | PGN/.pgn.zst → `data.npz` (Elo-filtered) |
| `scripts/train.py` | supervised training, loss = CE(policy) + MSE(value) |
| `app.py`, `templates/` | Flask API + single-page board (flip, undo, legal-move dots) |

Value labels: White win = 1, draw = 0, Black win = -1, stored from the mover's view (sign flipped on Black's turns).
Policy index = `from*64+to` (mirrored for Black); promotions default to queen.

## Run the app (iPad: use GitHub Codespaces)
Open the repo → **Code → Codespaces → Create**, then:
```bash
pip install -r requirements.txt
python app.py        # Codespaces pops up the forwarded port → opens in browser
```

## Train (free GPU: Colab or Kaggle, from your iPad browser)
1. Download a Lichess Elite Database month (database.nikonoel.fr) — or a TWIC / Caissabase PGN.
2. In a Colab/Kaggle notebook: `!git clone <your repo>` then
```bash
pip install -r requirements.txt
python scripts/prepare_data.py games.pgn.zst --min_elo 2400 --max_pos 500000
python scripts/train.py --epochs 5
```
3. Download `model.pt`, upload it to the repo root (or `MODEL_PATH=...`) and restart the app.

## Ideas to improve
Bigger net (`ChessNet(128, 10)`), more data, underpromotion moves, opening book, batching MCTS leaf evals,
self-play fine-tuning after supervised pretraining.
