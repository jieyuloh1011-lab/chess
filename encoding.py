"""Board/move encoding. Everything is from the SIDE-TO-MOVE's perspective
(board is mirrored when Black is to move), which makes the net colour-agnostic."""
import chess, numpy as np

N_PLANES, N_MOVES = 18, 64 * 64


def _sq(sq, turn):
    return sq if turn == chess.WHITE else sq ^ 56  # vertical mirror


def board_to_planes(board: chess.Board) -> np.ndarray:
    t = board.turn
    x = np.zeros((N_PLANES, 8, 8), np.float32)
    for sq, p in board.piece_map().items():
        s = _sq(sq, t)
        x[p.piece_type - 1 + (0 if p.color == t else 6), s // 8, s % 8] = 1
    x[12] = board.has_kingside_castling_rights(t)
    x[13] = board.has_queenside_castling_rights(t)
    x[14] = board.has_kingside_castling_rights(not t)
    x[15] = board.has_queenside_castling_rights(not t)
    if board.ep_square is not None:
        s = _sq(board.ep_square, t)
        x[16, s // 8, s % 8] = 1
    x[17] = min(board.halfmove_clock, 100) / 100
    return x


def move_to_index(move: chess.Move, turn: bool) -> int:
    return _sq(move.from_square, turn) * 64 + _sq(move.to_square, turn)


def legal_index_map(board: chess.Board) -> dict:
    """index -> Move. Promotions collapse to queen (underpromotion is rare)."""
    out = {}
    for m in board.legal_moves:
        if m.promotion and m.promotion != chess.QUEEN:
            continue
        out[move_to_index(m, board.turn)] = m
    return out
