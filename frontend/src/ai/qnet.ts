export type Weights = {
  w1: number[][];
  b1: number[];
  w2: number[][];
  b2: number[];
  w3: number[][];
  b3: number[];
};

function relu(x: number): number {
  return x > 0 ? x : 0;
}

function linear(input: number[], W: number[][], b: number[]): number[] {
  return W.map((row, i) =>
    row.reduce((sum, w, j) => sum + w * input[j], b[i]),
  );
}

export function forwardQ(
  weights: Weights,
  board: number[],
  player: number,
): number[] {
  const x = board.map((v) => v * player);
  let h = linear(x, weights.w1, weights.b1).map(relu);
  h = linear(h, weights.w2, weights.b2).map(relu);
  return linear(h, weights.w3, weights.b3);
}

export function aiMove(
  weights: Weights,
  board: number[],
  player: number,
): number {
  const q = forwardQ(weights, board, player);
  for (let i = 0; i < 9; i++) {
    if (board[i] !== 0) {
      q[i] = -1e9;
    }
  }
  return q.indexOf(Math.max(...q));
}

export async function loadWeights(): Promise<Weights> {
  const res = await fetch("/weights.json");
  if (!res.ok) {
    throw new Error("Failed to load weights.json");
  }
  return res.json() as Promise<Weights>;
}
