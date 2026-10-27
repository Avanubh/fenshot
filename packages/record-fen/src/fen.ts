/**
 * Model output to FEN piece placement. The classifier emits 13-class
 * probabilities per tile in A1..H8 rank order; class labels are
 * '1KQRBNPkqrbnp' (index 0 = empty square).
 *
 * Enhanced with:
 * - FIDE back-rank pawn repair
 * - Tall-piece cross-tile vertical spillover suppression
 * - FIDE legality scoring
 */

const LABELS = "1KQRBNPkqrbnp";

export interface RecognitionResult {
  /** FEN piece-placement field (ranks 8..1), run-length compressed. */
  placement: string;
  /** Per-tile argmax confidence, A1..H8 rank order. */
  confidences: number[];
  /** The weakest tile confidence. Below ~0.7 treat as unreliable. */
  minConfidence: number;
  /** Average tile confidence; used to arbitrate between candidate
   *  board alignments (recognize.ts). */
  meanConfidence: number;
}

export interface FideLegalityResult {
  legal: boolean;
  score: number;
  reason: string;
}

export function evaluateFideLegality(placement: string): FideLegalityResult {
  const rows = placement.split("/");
  if (rows.length !== 8) return { legal: false, score: -1000, reason: "invalid rank count" };

  let whiteKings = 0, blackKings = 0, whitePawns = 0, blackPawns = 0;
  let whiteTotal = 0, blackTotal = 0, invalidPawns = 0;
  let whiteKingPos: { rank: number; file: number } | null = null;
  let blackKingPos: { rank: number; file: number } | null = null;

  for (let r = 0; r < 8; r++) {
    const rank = 8 - r;
    let file = 0;
    for (const ch of rows[r]) {
      if (ch >= "1" && ch <= "8") {
        file += Number(ch);
      } else {
        if (ch === "K") { whiteKings++; whiteTotal++; whiteKingPos = { rank, file }; }
        else if (ch === "k") { blackKings++; blackTotal++; blackKingPos = { rank, file }; }
        else if (ch === "P") { whitePawns++; whiteTotal++; if (rank === 1 || rank === 8) invalidPawns++; }
        else if (ch === "p") { blackPawns++; blackTotal++; if (rank === 1 || rank === 8) invalidPawns++; }
        else if (ch >= "A" && ch <= "Z") { whiteTotal++; }
        else if (ch >= "a" && ch <= "z") { blackTotal++; }
        file++;
      }
    }
    if (file !== 8) return { legal: false, score: -1000, reason: "rank file width != 8" };
  }

  if (whiteKings !== 1 || blackKings !== 1) {
    return { legal: false, score: -500, reason: "Kings: W=" + whiteKings + ", B=" + blackKings };
  }

  if (whiteKingPos && blackKingPos) {
    const dr = Math.abs(whiteKingPos.rank - blackKingPos.rank);
    const df = Math.abs(whiteKingPos.file - blackKingPos.file);
    if (dr <= 1 && df <= 1) {
      return { legal: false, score: -300, reason: "Kings are adjacent" };
    }
  }

  if (invalidPawns > 0) {
    return { legal: false, score: -200 * invalidPawns, reason: invalidPawns + " pawns on rank 1 or 8" };
  }

  let penalty = 0;
  if (whitePawns > 8) penalty += (whitePawns - 8) * 60;
  if (blackPawns > 8) penalty += (blackPawns - 8) * 60;
  if (whiteTotal > 16) penalty += (whiteTotal - 16) * 60;
  if (blackTotal > 16) penalty += (blackTotal - 16) * 60;

  const score = 500 - penalty;
  return { legal: penalty === 0, score, reason: "ok" };
}

export function probsToPlacement(probs: Float32Array): RecognitionResult {
  const names: string[] = [];
  const confidences: number[] = [];
  const top2Indices: Array<{ bestIdx: number; secondIdx: number; bestProb: number; secondProb: number }> = [];

  for (let i = 0; i < 64; i++) {
    let best = -1;
    let bestIdx = 0;
    let second = -1;
    let secondIdx = 0;
    for (let c = 0; c < 13; c++) {
      const p = probs[i * 13 + c];
      if (p > best) {
        second = best;
        secondIdx = bestIdx;
        best = p;
        bestIdx = c;
      } else if (p > second) {
        second = p;
        secondIdx = c;
      }
    }
    names.push(LABELS[bestIdx]);
    confidences.push(best);
    top2Indices.push({ bestIdx, secondIdx, bestProb: best, secondProb: second });
  }

  const rawMinConfidence = Math.min(...confidences);

  // 1. Tall piece vertical spillover demarcation
  for (let r = 0; r < 7; r++) {
    for (let f = 0; f < 8; f++) {
      const tileBelow = r * 8 + f;
      const tileAbove = (r + 1) * 8 + f;
      const pieceBelow = names[tileBelow];
      const isTallBelow = "KQRBkqr".includes(pieceBelow) && confidences[tileBelow] > 0.80;

      if (isTallBelow) {
        const pieceAbove = names[tileAbove];
        const isPawnAbove = (pieceAbove === "P" || pieceAbove === "p") && confidences[tileAbove] < 0.65;
        const probEmpty = probs[tileAbove * 13 + 0];
        if (isPawnAbove && probEmpty > 0.15) {
          names[tileAbove] = "1";
          confidences[tileAbove] = probEmpty;
        }
      }
    }
  }

  // 2. FIDE Back-Rank Pawn Repair
  for (let f = 0; f < 8; f++) {
    const rank1Tile = 0 * 8 + f;
    if (names[rank1Tile] === "P" || names[rank1Tile] === "p") {
      const alt = top2Indices[rank1Tile];
      const altChar = LABELS[alt.secondIdx];
      names[rank1Tile] = (altChar === "P" || altChar === "p") ? "1" : altChar;
      confidences[rank1Tile] = alt.secondProb;
    }

    const rank8Tile = 7 * 8 + f;
    if (names[rank8Tile] === "P" || names[rank8Tile] === "p") {
      const alt = top2Indices[rank8Tile];
      const altChar = LABELS[alt.secondIdx];
      names[rank8Tile] = (altChar === "P" || altChar === "p") ? (names[rank8Tile] === "P" ? "Q" : "q") : altChar;
      confidences[rank8Tile] = alt.secondProb;
    }
  }

  const ranks: string[] = [];
  for (let r = 7; r >= 0; r--) {
    ranks.push(names.slice(r * 8, r * 8 + 8).join(""));
  }
  const placement = ranks
    .join("/")
    .replace(/1{2,}/g, (m) => String(m.length));
  return {
    placement,
    confidences,
    minConfidence: rawMinConfidence,
    meanConfidence: confidences.reduce((a, b) => a + b, 0) / confidences.length,
  };
}

/** Mirror a placement for a board photographed with Black at the
 *  bottom (rotate 180 degrees: reverse ranks and files). */
export function flipPlacement(placement: string): string {
  return placement
    .split("/")
    .reverse()
    .map((rank) => rank.split("").reverse().join(""))
    .join("/");
}

/** Mean rank of each color's pawns, reading `placement` as a
 *  white-at-bottom board (rank 8 = first row, rank 1 = last row).
 *  Returns null for a color with no pawns. */
export function meanPawnRanks(placement: string): {
  white: number | null;
  black: number | null;
} {
  const ranks = placement.split("/");
  const white: number[] = [];
  const black: number[] = [];
  ranks.forEach((row, i) => {
    const rank = 8 - i;
    for (const ch of row) {
      if (ch === "P") white.push(rank);
      else if (ch === "p") black.push(rank);
    }
  });
  const mean = (xs: number[]) =>
    xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
  return { white: mean(white), black: mean(black) };
}

/** Decide board orientation from pawn-advance direction. */
export function resolveOrientation(placement: string): {
  placement: string;
  orientation: "white" | "black";
} {
  const naturalness = (p: string): number | null => {
    const { white, black } = meanPawnRanks(p);
    if (white == null || black == null) return null;
    return black - white;
  };

  const asRead = naturalness(placement);
  const rotated = flipPlacement(placement);
  const asRotated = naturalness(rotated);

  if (asRead != null && asRotated != null && asRotated > asRead) {
    return { placement: rotated, orientation: "black" };
  }
  return { placement, orientation: "white" };
}
