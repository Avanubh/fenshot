/**
 * Tile extraction for the chess-tiles CNN. Port of getChessTilesGray
 * from Elucidation/tensorflow_chessbot (MIT).
 *
 * Enhanced with Paul Heckbert (1989) Projective Homography for
 * perspective quadrilateral rectification.
 */

import { type BoardCorners, type GrayImage, createHomographyMap } from "./detect";

const BOARD_PX = 256;
const TILE_PX = 32;
export const TILE_INPUT_SIZE = TILE_PX * TILE_PX;

/** Sample with edge clamping (np.pad mode="edge" equivalent). */
function sampleClamped(img: GrayImage, x: number, y: number): number {
  const cx = Math.min(img.width - 1, Math.max(0, x));
  const cy = Math.min(img.height - 1, Math.max(0, y));
  return img.data[cy * img.width + cx];
}

/** Bilinear resize of the corners crop to 256x256 through Homography, values in [0,1]. */
export function extractBoardImage(img: GrayImage, corners: BoardCorners): Float32Array {
  let mapPoint: (u: number, v: number) => { x: number; y: number };

  if (corners && corners.quad && corners.quad.length === 4) {
    mapPoint = createHomographyMap(corners.quad);
  } else {
    const { x0, y0, x1, y1 } = corners;
    const cw = x1 - x0;
    const ch = y1 - y0;
    mapPoint = (u, v) => ({ x: x0 + u * cw, y: y0 + v * ch });
  }

  const out = new Float32Array(BOARD_PX * BOARD_PX);
  for (let ty = 0; ty < BOARD_PX; ty++) {
    const v = (ty + 0.5) / BOARD_PX;
    for (let tx = 0; tx < BOARD_PX; tx++) {
      const u = (tx + 0.5) / BOARD_PX;
      const pt = mapPoint(u, v);
      const sx = pt.x - 0.5;
      const sy = pt.y - 0.5;
      const fx = Math.floor(sx);
      const fy = Math.floor(sy);
      const wx = sx - fx;
      const wy = sy - fy;
      const val =
        sampleClamped(img, fx, fy) * (1 - wx) * (1 - wy) +
        sampleClamped(img, fx + 1, fy) * wx * (1 - wy) +
        sampleClamped(img, fx, fy + 1) * (1 - wx) * wy +
        sampleClamped(img, fx + 1, fy + 1) * wx * wy;
      out[ty * BOARD_PX + tx] = val / 255;
    }
  }
  return out;
}

/** Slice the 256x256 board into the [64, 1024] model input. */
export function boardToTiles(board: Float32Array): Float32Array {
  const out = new Float32Array(64 * TILE_INPUT_SIZE);
  for (let rank = 0; rank < 8; rank++) {
    for (let file = 0; file < 8; file++) {
      const tile = rank * 8 + file;
      const srcY0 = (7 - rank) * TILE_PX;
      const srcX0 = file * TILE_PX;
      for (let y = 0; y < TILE_PX; y++) {
        const srcRow = (srcY0 + y) * BOARD_PX + srcX0;
        const dstRow = tile * TILE_INPUT_SIZE + y * TILE_PX;
        for (let x = 0; x < TILE_PX; x++) {
          out[dstRow + x] = board[srcRow + x];
        }
      }
    }
  }
  return out;
}

export function extractTiles(img: GrayImage, corners: BoardCorners): Float32Array {
  return boardToTiles(extractBoardImage(img, corners));
}

/** RGBA ImageData bytes to 0-255 grayscale (ITU-R 601 luma). */
export function rgbaToGray(rgba: Uint8ClampedArray, width: number, height: number): GrayImage {
  const data = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const off = i * 4;
    data[i] = 0.299 * rgba[off] + 0.587 * rgba[off + 1] + 0.114 * rgba[off + 2];
  }
  return { data, width, height };
}
