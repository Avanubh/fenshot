# fenshot

**Screenshot in. FEN out. State-of-the-art chessboard recognition.**

[![CI](https://github.com/Avanubh/fenshot/actions/workflows/ci.yml/badge.svg)](https://github.com/Avanubh/fenshot/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/%40avanubh%2Ffenshot)](https://www.npmjs.com/package/@avanubh/fenshot)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Paste any chessboard screenshot, a chess.com game, a Lichess puzzle, a diagram from a chess book, a mobile camera photo, or a position from Reddit, and get the position as a valid FEN, entirely on your device. No account, no upload, zero server latency.

Repository: **[github.com/Avanubh/fenshot](https://github.com/Avanubh/fenshot)**

---

## 10/10 Vision Engine Highlights (v1.0.0)

- **Paul Heckbert Projective Homography ($H \in \text{PGL}(3, \mathbb{R})$)**: Transforms angled camera captures and trapezoidal table tilts into rectified $256 \times 256$ square boards via closed-form projective mappings.
- **Cross-Tile Vertical Spillover Demarcation**: Suppresses false pawn detections caused by tall piece crowns, crosses, and miters protruding into the rank above.
- **FIDE Rules & Legality Matrix**: Strictly verifies King counts ($K=1, k=1$), king-to-king separation, and piece limits. Automatically repairs impossible back-rank pawns (rank 1 or 8) into legal pieces or empty squares.
- **Multi-Hypothesis Bayesian Arbitration**: Evaluates candidates across Hough sequences, 2D directional energy envelopes ($E(x,y) = |G_x| \cdot |G_y|$), and perspective quads with an objective matrix.

---

## How it works

| Your screenshot | | What fenshot reads |
|:---:|:---:|:---:|
| <img src="packages/fenshot/tests/fixtures/lichess-italian-black.png" width="360" alt="Lichess screenshot from Black's point of view" /> | **→** | `r1bqk1nr/pppp1ppp/2n5/`<br/>`2b1p3/2B1P3/5N2/`<br/>`PPPP1PPP/RNBQK2R` |

The screenshot above is from **Black's point of view**; fenshot detects that from pawn-advance direction and flips the read automatically.

Under the hood:

1. **Board detection** finds the board via 1D gradient peak analysis and 2D directional gradient energy tensors.
2. **Homography Rectification** solves the 8-DOF projective transformation mapping 4-corner perspective trapezoids into canonical squares.
3. **Tile classification** reads the 64 squares with an optimized CNN (~330k params, 1.3 MB ONNX) running locally via onnxruntime-web WASM.
4. **Spatial Context & Spillover Filtering** removes false piece detections on empty squares above tall pieces.
5. **FIDE Legality Arbitration** guarantees returned positions adhere to physical chess rules.

---

## Use it as a library

```bash
npm install @avanubh/fenshot onnxruntime-web
```

```ts
import { createRecognizer, resolveOrientation, placementToFen } from "@avanubh/fenshot";

const recognizer = createRecognizer({
  modelUrl: "/models/chess-tiles-v2.onnx",
  wasmPaths: "/ort/",
});

const result = await recognizer.recognize(file);
if (result?.reliable) {
  const { placement } = resolveOrientation(result.placement);
  console.log(placementToFen(placement, "w"));
}
```

Full API docs and bundler specifics: [packages/fenshot/README.md](packages/fenshot/README.md).

---

## Repo layout

| Path | What |
|------|------|
| `packages/fenshot` | The npm package (`@avanubh/fenshot`): detection, homography, CNN classification, FEN composition |
| `apps/extension` | Browser extension (Chrome + Firefox + Edge): one-click scan of the visible page |
| `apps/web` | Demo app (Vite + React), deployed to GitHub Pages |
| `tools/tile-classifier` | Training pipeline and evaluation gates |

---

## Development & Testing

```bash
npm install
npm test              # runs all 122 unit and regression tests
npm run build         # builds @avanubh/fenshot
npm run dev           # runs the web demo
```

---

## Credits & License

- Original baseline & concept inspired by [Elucidation/tensorflow_chessbot](https://github.com/Elucidation/tensorflow_chessbot) and [scoriiu/fenshot](https://github.com/scoriiu/fenshot).
- Enhanced, re-architected, and maintained by **[Anubhav](https://github.com/Avanubh)**.
- Licensed under the [MIT License](LICENSE).
