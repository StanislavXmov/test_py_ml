import type { Board as BoardState } from "../game/board";
import { AI, cellLabel, HUMAN } from "../game/board";

type BoardProps = {
  board: BoardState;
  disabled: boolean;
  onCellClick: (index: number) => void;
};

export function Board({ board, disabled, onCellClick }: BoardProps) {
  return (
    <div className="board" role="grid" aria-label="Игровое поле">
      {board.map((cell, index) => (
        <button
          key={index}
          type="button"
          className={[
            "cell",
            cell === HUMAN ? "cell--x" : "",
            cell === AI ? "cell--o" : "",
          ]
            .filter(Boolean)
            .join(" ")}
          disabled={disabled || cell !== 0}
          aria-label={`Клетка ${index}`}
          onClick={() => onCellClick(index)}
        >
          {cellLabel(cell)}
        </button>
      ))}
    </div>
  );
}
