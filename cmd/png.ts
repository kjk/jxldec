// png.ts -- minimal PNG decoder, used to turn libjxl's testdata PNGs into the
// PNM/PAM that cjxl reads natively.
//
//   bun cmd/png.ts in.png out.pam
//
// We deliberately do not build libpng into the libjxl oracle (its cmake glue
// can't find the bundled zlib's generated zconf.h), and cjxl/djxl handle
// PNM/PAM without it -- so the only thing missing is a PNG reader, and 200
// lines of TypeScript is cheaper than fighting the build.
//
// Supports 8- and 16-bit gray / gray+alpha / RGB / RGBA / palette, both
// non-interlaced and Adam7.
import { inflateSync } from "zlib";
import { readFileSync, writeFileSync } from "fs";

export type PngImage = {
  width: number;
  height: number;
  channels: number; // 1, 2, 3 or 4
  depth: 8 | 16;
  data: Uint8Array; // big-endian samples, channels interleaved
};

const CHANNELS: Record<number, number> = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

/** Reverses the per-scanline filters of one (sub)image. */
function unfilter(
  raw: Uint8Array,
  width: number,
  height: number,
  bpp: number,
  bytesPerRow: number,
): Uint8Array {
  const out = new Uint8Array(bytesPerRow * height);
  let pos = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[pos++];
    const row = out.subarray(y * bytesPerRow, (y + 1) * bytesPerRow);
    const prev = y > 0 ? out.subarray((y - 1) * bytesPerRow, y * bytesPerRow) : null;
    for (let x = 0; x < bytesPerRow; x++) {
      const cur = raw[pos + x];
      const a = x >= bpp ? row[x - bpp] : 0;
      const b = prev ? prev[x] : 0;
      const c = prev && x >= bpp ? prev[x - bpp] : 0;
      let v: number;
      switch (filter) {
        case 0: v = cur; break;
        case 1: v = cur + a; break;
        case 2: v = cur + b; break;
        case 3: v = cur + ((a + b) >> 1); break;
        case 4: v = cur + paeth(a, b, c); break;
        default: throw new Error(`bad PNG filter ${filter}`);
      }
      row[x] = v & 0xff;
    }
    pos += bytesPerRow;
  }
  return out;
}

const A7_XOFF = [0, 4, 0, 2, 0, 1, 0];
const A7_YOFF = [0, 0, 4, 0, 2, 0, 1];
const A7_XSTEP = [8, 8, 4, 4, 2, 2, 1];
const A7_YSTEP = [8, 8, 8, 4, 4, 2, 2];

export function decodePng(buf: Uint8Array): PngImage {
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  for (let i = 0; i < 8; i++) {
    if (buf[i] !== sig[i]) throw new Error("not a PNG");
  }
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let pos = 8;
  let width = 0, height = 0, depth = 0, colorType = 0, interlace = 0;
  let palette: Uint8Array | null = null;
  let trns: Uint8Array | null = null;
  const idat: Uint8Array[] = [];

  while (pos + 8 <= buf.length) {
    const len = dv.getUint32(pos);
    const type = String.fromCharCode(buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]);
    const body = buf.subarray(pos + 8, pos + 8 + len);
    if (type === "IHDR") {
      width = dv.getUint32(pos + 8);
      height = dv.getUint32(pos + 12);
      depth = buf[pos + 16];
      colorType = buf[pos + 17];
      interlace = buf[pos + 20];
    } else if (type === "PLTE") {
      palette = new Uint8Array(body);
    } else if (type === "tRNS") {
      trns = new Uint8Array(body);
    } else if (type === "IDAT") {
      idat.push(new Uint8Array(body));
    } else if (type === "IEND") {
      break;
    }
    pos += 12 + len;
  }
  if (!(colorType in CHANNELS)) throw new Error(`unsupported PNG color type ${colorType}`);

  const merged = new Uint8Array(idat.reduce((n, c) => n + c.length, 0));
  let off = 0;
  for (const c of idat) {
    merged.set(c, off);
    off += c.length;
  }
  const raw = new Uint8Array(inflateSync(merged));

  const srcChannels = CHANNELS[colorType];
  const bitsPerPixel = srcChannels * depth;
  const bpp = Math.max(1, bitsPerPixel >> 3);

  // Expand into a byte-per-sample (or 2 bytes at depth 16) plane.
  const sampleBytes = depth === 16 ? 2 : 1;
  const planar = new Uint8Array(width * height * srcChannels * sampleBytes);

  const placePass = (
    img: Uint8Array,
    pw: number,
    ph: number,
    xoff: number,
    xstep: number,
    yoff: number,
    ystep: number,
  ) => {
    const rowBytes = Math.ceil((pw * bitsPerPixel) / 8);
    for (let y = 0; y < ph; y++) {
      const row = img.subarray(y * rowBytes, (y + 1) * rowBytes);
      for (let x = 0; x < pw; x++) {
        const dstX = xoff + x * xstep;
        const dstY = yoff + y * ystep;
        const dst = (dstY * width + dstX) * srcChannels * sampleBytes;
        for (let c = 0; c < srcChannels; c++) {
          if (depth === 16) {
            const s = (x * srcChannels + c) * 2;
            planar[dst + c * 2] = row[s];
            planar[dst + c * 2 + 1] = row[s + 1];
          } else if (depth === 8) {
            planar[dst + c] = row[x * srcChannels + c];
          } else {
            // 1/2/4-bit: one channel only (gray or palette index)
            const bitPos = x * depth;
            const byte = row[bitPos >> 3];
            const shift = 8 - depth - (bitPos & 7);
            planar[dst + c] = (byte >> shift) & ((1 << depth) - 1);
          }
        }
      }
    }
  };

  if (interlace === 0) {
    const rowBytes = Math.ceil((width * bitsPerPixel) / 8);
    const img = unfilter(raw, width, height, bpp, rowBytes);
    placePass(img, width, height, 0, 1, 0, 1);
  } else {
    let rpos = 0;
    for (let p = 0; p < 7; p++) {
      const pw = Math.ceil((width - A7_XOFF[p]) / A7_XSTEP[p]);
      const ph = Math.ceil((height - A7_YOFF[p]) / A7_YSTEP[p]);
      if (pw <= 0 || ph <= 0) continue;
      const rowBytes = Math.ceil((pw * bitsPerPixel) / 8);
      const chunk = raw.subarray(rpos, rpos + (rowBytes + 1) * ph);
      rpos += (rowBytes + 1) * ph;
      const img = unfilter(chunk, pw, ph, bpp, rowBytes);
      placePass(img, pw, ph, A7_XOFF[p], A7_XSTEP[p], A7_YOFF[p], A7_YSTEP[p]);
    }
  }

  // Palette expansion (always to 8-bit RGB / RGBA).
  if (colorType === 3) {
    if (!palette) throw new Error("indexed PNG without PLTE");
    const hasAlpha = !!trns;
    const outCh = hasAlpha ? 4 : 3;
    const out = new Uint8Array(width * height * outCh);
    for (let i = 0; i < width * height; i++) {
      const idx = planar[i];
      out[i * outCh] = palette[idx * 3];
      out[i * outCh + 1] = palette[idx * 3 + 1];
      out[i * outCh + 2] = palette[idx * 3 + 2];
      if (hasAlpha) out[i * outCh + 3] = idx < trns!.length ? trns![idx] : 255;
    }
    return { width, height, channels: outCh, depth: 8, data: out };
  }

  // Sub-byte gray: scale up to 8 bits.
  if (depth < 8) {
    const max = (1 << depth) - 1;
    for (let i = 0; i < planar.length; i++) {
      planar[i] = Math.round((planar[i] * 255) / max);
    }
  }

  return {
    width,
    height,
    channels: srcChannels,
    depth: depth === 16 ? 16 : 8,
    data: planar,
  };
}

const TUPLE = ["", "GRAYSCALE", "GRAYSCALE_ALPHA", "RGB", "RGB_ALPHA"];

export function encodePam(img: PngImage): Uint8Array {
  const maxval = img.depth === 16 ? 65535 : 255;
  const header =
    `P7\nWIDTH ${img.width}\nHEIGHT ${img.height}\nDEPTH ${img.channels}\n` +
    `MAXVAL ${maxval}\nTUPLTYPE ${TUPLE[img.channels]}\nENDHDR\n`;
  const head = new TextEncoder().encode(header);
  const out = new Uint8Array(head.length + img.data.length);
  out.set(head, 0);
  out.set(img.data, head.length);
  return out;
}

export function pngToPam(pngPath: string, pamPath: string): PngImage {
  const img = decodePng(new Uint8Array(readFileSync(pngPath)));
  writeFileSync(pamPath, encodePam(img));
  return img;
}

if (import.meta.main) {
  const [inPath, outPath] = process.argv.slice(2);
  if (!inPath || !outPath) {
    console.log("usage: bun cmd/png.ts in.png out.pam");
    process.exit(2);
  }
  const img = pngToPam(inPath, outPath);
  console.log(`${inPath}: ${img.width}x${img.height}, ${img.channels}ch, ${img.depth}-bit`);
}
