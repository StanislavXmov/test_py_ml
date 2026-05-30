import { useCallback, useEffect, useState } from "react";

import { aiMove, loadWeights, type Weights } from "./ai/qnet";
import { Board } from "./components/Board";
import {
  AI,
  createEmptyBoard,
  HUMAN,
  resultMessage,
  winner,
  type Board as BoardState,
  type GameResult,
} from "./game/board";
import "./App.css";

export default function App() {
  const [weights, setWeights] = useState<Weights | null>(null);
  const [board, setBoard] = useState<BoardState>(createEmptyBoard);
  const [result, setResult] = useState<GameResult>(null);
  const [lastAiMove, setLastAiMove] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadWeights()
      .then(setWeights)
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Unknown error");
      })
      .finally(() => setLoading(false));
  }, []);

  const resetGame = useCallback(() => {
    setBoard(createEmptyBoard());
    setResult(null);
    setLastAiMove(null);
  }, []);

  const handleCellClick = useCallback(
    (index: number) => {
      if (!weights || result !== null || board[index] !== 0) {
        return;
      }

      const nextBoard = [...board];
      nextBoard[index] = HUMAN;

      const humanResult = winner(nextBoard);
      if (humanResult !== null) {
        setBoard(nextBoard);
        setResult(humanResult);
        return;
      }

      const move = aiMove(weights, nextBoard, AI);
      nextBoard[move] = AI;

      setBoard(nextBoard);
      setLastAiMove(move);
      setResult(winner(nextBoard));
    },
    [board, result, weights],
  );

  if (loading) {
    return (
      <main className="app">
        <p className="status">Загрузка модели...</p>
      </main>
    );
  }

  if (error || !weights) {
    return (
      <main className="app">
        <p className="status error">
          {error ?? "Не удалось загрузить weights.json"}
        </p>
      </main>
    );
  }

  return (
    <main className="app">
      <h1>Крестики-нолики</h1>
      <p className="subtitle">Вы — X, ИИ — O. Всё работает на клиенте.</p>

      <Board
        board={board}
        disabled={result !== null}
        onCellClick={handleCellClick}
      />

      <div className="info">
        {result !== null ? (
          <p className="result">{resultMessage(result)}</p>
        ) : (
          <p className="hint">Ваш ход — выберите клетку</p>
        )}

        {lastAiMove !== null && result === null && (
          <p className="hint">ИИ ходит в клетку {lastAiMove}</p>
        )}

        <button type="button" className="reset" onClick={resetGame}>
          Новая игра
        </button>
      </div>
    </main>
  );
}
