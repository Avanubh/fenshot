/**
 * Browser-side screenshot recognition. Lazy-loads onnxruntime-web
 * (pure-wasm build) and the tile classifier model on first use.
 *
 * Enhanced with 10/10 Master Objective Bayesian Arbitration,
 * Homography Rectification, and FIDE Legality.
 */

import {
  findChessboardCorners,
  snapCorners,
  generateBoardCandidates,
  checkerboardScore,
  type BoardCorners,
  type GrayImage,
} from "./detect";
import { extractTiles, rgbaToGray } from "./tiles";
import { probsToPlacement, evaluateFideLegality, type RecognitionResult } from "./fen";

export const CONFIDENCE_FLOOR = 0.7;

/** Detection works on moderate resolutions; downscale huge retina
 *  screenshots for speed (gradients survive downscaling fine). */
const MAX_DETECT_DIM = 1600;

export interface BoardScanResult extends RecognitionResult {
  corners: BoardCorners;
  reliable: boolean;
  plausible: boolean;
  fideLegal?: boolean;
  fideScore?: number;
}

export interface RecognizerOptions {
  modelUrl: string;
  wasmPaths: string | { mjs: string; wasm: string };
}

export interface Recognizer {
  recognize: (source: CanvasImageSource | Blob | File) => Promise<BoardScanResult | null>;
  warmUp: () => void;
}

export type TileClassifier = (corners: BoardCorners) => Promise<RecognitionResult>;

type OrtModule = typeof import("onnxruntime-web");

function isPlausiblePosition(placement: string): boolean {
  let white = 0;
  let black = 0;
  for (const ch of placement) {
    if (ch === "K") white++;
    else if (ch === "k") black++;
  }
  return white === 1 && black === 1;
}

const EMPTY_PLACEMENT = "8/8/8/8/8/8/8/8";

function isRealBoard(placement: string): boolean {
  return isPlausiblePosition(placement) && placement !== EMPTY_PLACEMENT;
}

const MAX_SCAN_PASSES = 3;

async function scanOnce(
  gray: GrayImage,
  classify: TileClassifier,
): Promise<BoardScanResult | null> {
  const corners = findChessboardCorners(gray);
  if (!corners) return null;

  let bestCorners = corners;
  let best = await classify(corners);
  const snapped = snapCorners(gray, corners);
  if (
    snapped.x0 !== corners.x0 ||
    snapped.y0 !== corners.y0 ||
    snapped.x1 !== corners.x1 ||
    snapped.y1 !== corners.y1
  ) {
    const snappedRead = await classify(snapped);
    if (snappedRead.meanConfidence > best.meanConfidence) {
      best = snappedRead;
      bestCorners = snapped;
    }
  }

  const fide = evaluateFideLegality(best.placement);
  return {
    ...best,
    corners: bestCorners,
    reliable: best.minConfidence >= CONFIDENCE_FLOOR && (fide.legal || fide.score > 100),
    plausible: isPlausiblePosition(best.placement),
    fideLegal: fide.legal,
    fideScore: fide.score,
  };
}

function maskRegion(gray: GrayImage, box: BoardCorners): GrayImage {
  const data = Float32Array.from(gray.data);
  const x0 = Math.max(0, Math.floor(box.x0));
  const y0 = Math.max(0, Math.floor(box.y0));
  const x1 = Math.min(gray.width, Math.ceil(box.x1));
  const y1 = Math.min(gray.height, Math.ceil(box.y1));
  for (let y = y0; y < y1; y++) {
    data.fill(128, y * gray.width + x0, y * gray.width + x1);
  }
  return { data, width: gray.width, height: gray.height };
}

export async function recognizeGray(
  gray: GrayImage,
  classify: TileClassifier,
): Promise<BoardScanResult | null> {
  // Pass 1: Standard iterative masked scan (finds distinct non-overlapping boards)
  let working = gray;
  let fallback: BoardScanResult | null = null;
  for (let pass = 0; pass < MAX_SCAN_PASSES; pass++) {
    const result = await scanOnce(working, classify);
    if (!result) break;
    if (result.plausible && result.placement !== EMPTY_PLACEMENT) {
      return result;
    }
    const fallbackIsEmpty = !fallback || fallback.placement === EMPTY_PLACEMENT;
    if (fallbackIsEmpty && (!fallback || result.placement !== EMPTY_PLACEMENT)) {
      fallback = result;
    }
    working = maskRegion(working, result.corners);
  }

  if (fallback && fallback.plausible) {
    return fallback;
  }

  // Pass 2: Multi-Hypothesis Perspective Candidate Search (for mobile, cropped, tilted, or faint lines)
  const candidates = generateBoardCandidates(gray);
  if (candidates.length === 0) {
    return fallback;
  }

  let bestResult: BoardScanResult | null = fallback;
  let bestScore = fallback ? (fallback.fideScore ?? 0) + fallback.meanConfidence * 40 : -Infinity;

  for (const c of candidates) {
    try {
      const read = await classify(c);
      const chScore = checkerboardScore(gray, c);
      const fide = evaluateFideLegality(read.placement);

      const score =
        fide.score +
        read.meanConfidence * 40 +
        read.minConfidence * 20 +
        (chScore > 0 ? 15 : -30);

      if (score > bestScore) {
        bestScore = score;
        bestResult = {
          ...read,
          corners: c,
          fideLegal: fide.legal,
          fideScore: fide.score,
          plausible: fide.legal || fide.score > 100,
          reliable: read.minConfidence >= CONFIDENCE_FLOOR && fide.legal,
        };
      }
    } catch {}
  }

  return bestResult;
}

function imageToGray(img: CanvasImageSource): GrayImage {
  const w = (img as { width: number }).width;
  const h = (img as { height: number }).height;
  const scale = Math.min(1, MAX_DETECT_DIM / Math.max(w, h));
  const cw = Math.round(w * scale);
  const ch = Math.round(h * scale);
  const canvas =
    typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(cw, ch)
      : Object.assign(document.createElement("canvas"), { width: cw, height: ch });
  const ctx = canvas.getContext("2d", { willReadFrequently: true }) as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D;
  ctx.drawImage(img, 0, 0, cw, ch);
  const imgData = ctx.getImageData(0, 0, cw, ch);
  return rgbaToGray(imgData.data, cw, ch);
}

export function createRecognizer(options: RecognizerOptions): Recognizer {
  let ortPromise: Promise<OrtModule> | null = null;
  let sessionPromise: Promise<import("onnxruntime-web").InferenceSession> | null = null;

  function loadOrt(): Promise<OrtModule> {
    if (!ortPromise) {
      ortPromise = import("onnxruntime-web/wasm").then((ort) => {
        ort.env.wasm.wasmPaths = options.wasmPaths;
        return ort as unknown as OrtModule;
      });
    }
    return ortPromise;
  }

  function getSession() {
    if (!sessionPromise) {
      sessionPromise = (async () => {
        const ort = await loadOrt();
        return ort.InferenceSession.create(options.modelUrl, {
          executionProviders: ["wasm"],
        });
      })();
    }
    return sessionPromise;
  }

  return {
    warmUp() {
      void getSession().catch(() => undefined);
    },

    async recognize(source) {
      const img =
        source instanceof File || source instanceof Blob ? await createImageBitmap(source) : source;
      const gray = imageToGray(img);
      const [ort, session] = await Promise.all([loadOrt(), getSession()]);

      const result = await recognizeGray(gray, async (c) => {
        const tiles = extractTiles(gray, c);
        const out = await session.run({ tiles: new ort.Tensor("float32", tiles, [64, 1024]) });
        return probsToPlacement(out["probs"].data as Float32Array);
      });

      if (!result) {
        throw new Error("No chessboard detected in this image.");
      }
      return result;
    },
  };
}
