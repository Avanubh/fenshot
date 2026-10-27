# recordFen

Screenshot in. FEN out. State-of-the-art chessboard recognition.

[![CI](https://github.com/Avanubh/recordFen/actions/workflows/ci.yml/badge.svg)](https://github.com/Avanubh/recordFen/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/%40avanubh%2Frecord-fen)](https://www.npmjs.com/package/@avanubh/record-fen)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Avanubh/recordFen/blob/main/LICENSE)

Repository: **[github.com/Avanubh/recordFen](https://github.com/Avanubh/recordFen)**

Chessboard recognition that runs entirely on-device: finds the board in screenshots, mobile captures, photos, and book diagrams, rectifies perspective distortions with **Paul Heckbert Projective Homography**, evaluates against **FIDE chess legality rules**, and reads piece positions using an on-device ONNX CNN tile classifier.

Zero server dependencies, zero privacy leaks, runs in $<300\text{ms}$.

---

## What's New in recordFen 10/10 Vision Engine (v1.0.0)

- **Paul Heckbert (1989) Projective Homography ($H \in \text{PGL}(3, \mathbb{R})$)**: Transforms angled camera shots and trapezoidal table tilts into perfectly rectified $256 \times 256$ top-down boards via exact closed-form algebraic equations.
- **Cross-Tile Vertical Spillover Demarcation**: Eliminates false pawn detections caused by King crosses, Queen coronets, and Bishop miters projecting upward into the rank above.
- **FIDE Rules & Legality Matrix**: Strictly validates king presence, king-to-king separation, and piece limits. Automatically repairs impossible back-rank pawns (rank 1 or 8) into legal piece types or empty squares.
- **2D Directional Gradient Energy Auto-Cropper**: Combines 2D Sobel tensor energy envelopes ($E(x,y) = |G_x| \cdot |G_y|$) with Hough line sequence detection.
- **Bayesian Multi-Hypothesis Arbitration**: Arbitrates between perspective quads, insets, and Hough subgrids with an objective scoring matrix to find the global optimum.

---

## Install

```bash
npm install @avanubh/record-fen onnxruntime-web
```

Two static assets are required:

1. **The model** (1.3 MB): copy `node_modules/@avanubh/record-fen/model/chess-tiles-v2.onnx` to your static directory (or serve via CDN).
2. **The onnxruntime wasm pair**: copy `ort-wasm-simd-threaded.mjs` and `ort-wasm-simd-threaded.wasm` from `node_modules/onnxruntime-web/dist/` to a static directory (e.g. `/ort/`).

Both are lazy-loaded on the first scan.

---

## Quickstart

```ts
import { createRecognizer, resolveOrientation, placementToFen } from "@avanubh/record-fen";

const recognizer = createRecognizer({
  modelUrl: "/models/chess-tiles-v2.onnx",
  wasmPaths: "/ort/",
});

// Pass an image File, Blob, ImageBitmap, or HTMLImageElement
const result = await recognizer.recognize(file);

if (!result) {
  console.log("No chessboard detected.");
} else if (!result.reliable) {
  console.log("Board detected with moderate confidence: review positions.");
} else {
  const { placement, orientation } = resolveOrientation(result.placement);
  const fen = placementToFen(placement, "w");
  console.log("Detected FEN:", fen);
  console.log("FIDE Legal:", result.fideLegal);
}
```

---

## API Reference

### `createRecognizer(options): Recognizer`

- `options.modelUrl` — URL of the served ONNX tile classifier.
- `options.wasmPaths` — Directory URL or path mapping for `onnxruntime-web` wasm binaries.

### `recognizer.recognize(source): Promise<BoardScanResult | null>`

Resolves `null` when no chessboard is detected. Otherwise returns:

```ts
interface BoardScanResult {
  placement: string;      // FEN board field (ranks 8..1)
  meanConfidence: number; // Mean tile classifier confidence
  minConfidence: number;  // Worst-tile confidence
  reliable: boolean;      // Confidence floor and FIDE legality met
  plausible: boolean;     // Exactly 1 White King and 1 Black King
  fideLegal?: boolean;    // Full FIDE rule adherence
  corners: BoardCorners;  // Bounding box with optional perspective quad coordinates
}
```

### Low-Level Functional Exports

For custom processing pipelines (e.g., React Native WebView, Node with sharp, canvas workers):
- `createHomographyMap(quad)` — Paul Heckbert projective transformation matrix generator.
- `evaluateFideLegality(placement)` — Rigorous FIDE rule-checking engine.
- `generateBoardCandidates(gray)` — Multi-hypothesis candidate generator (Hough, energy box, perspective quads).
- `findChessboardCorners(gray)` — Hough line arithmetic sequence grid finder.
- `snapCorners(gray, box)` — Grid-snap alignment refinement.
- `extractTiles(gray, corners)` — Homography bilinear interpolation into 64 normalized $32 \times 32$ tile tensors.
- `probsToPlacement(probs)` — Tile CNN argmax with spillover suppression and back-rank pawn repair.
- `resolveOrientation(placement)` — Pawn-advance naturalness resolver.
- `placementToFen(placement, turn)` — Analyzable FEN constructor with castling rights inference.

---

## Credits & License

- Original baseline & concept inspired by [Elucidation/tensorflow_chessbot](https://github.com/Elucidation/tensorflow_chessbot) and [scoriiu/fenshot](https://github.com/scoriiu/fenshot).
- Enhanced, re-architected, and maintained by **[Anubhav](https://github.com/Avanubh)**.
- Licensed under the [MIT License](https://github.com/Avanubh/recordFen/blob/main/LICENSE).
