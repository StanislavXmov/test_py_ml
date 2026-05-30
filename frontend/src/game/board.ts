export const HUMAN = 1;
export const AI = -1;

export type Cell = -1 | 0 | 1;
export type Board = Cell[];
export type GameResult = -1 | 0 | 1 | null;

const LINES = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8],
  [2, 4, 6],
] as const;

export function createEmptyBoard(): Board {
  return Array.from({ length: 9 }, () => 0);
}

export function winner(board: Board): GameResult {
  for (const [a, b, c] of LINES) {
    const sum = board[a] + board[b] + board[c];
    if (sum === 3) {
      return 1;
    }
    if (sum === -3) {
      return -1;
    }
  }

  if (!board.includes(0)) {
    return 0;
  }

  return null;
}

export function cellLabel(cell: Cell): string {
  if (cell === HUMAN) {
    return "X";
  }
  if (cell === AI) {
    return "O";
  }
  return "";
}

export function resultMessage(result: Exclude<GameResult, null>): string {
  if (result === HUMAN) {
    return "Вы победили!";
  }
  if (result === AI) {
    return "Победил ИИ!";
  }
  return "Ничья!";
}
