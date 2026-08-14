// Tests the viewer's PLY parser against files the pipeline actually writes.
//
// The parser is the riskiest part of the viewer: a wrong stride or a
// misinterpreted colour encoding produces a plausible-looking cloud that is
// subtly wrong, which is far harder to notice than a blank screen. So it is
// extracted from the HTML and exercised directly, in node, against real bytes.
//
//   node tools/test_viewer.mjs
//
// Requires the fixtures written by tools/make_viewer_fixtures.py.

import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const results = [];
const check = (label, cond) => results.push([label, !!cond]);

// Pull the parser out of the page. The banner comments are stable delimiters
// that exist precisely so this extraction does not depend on line numbers.
const html = readFileSync(join(ROOT, "viewer", "walkthrough.html"), "utf8");
const start = html.indexOf("const SH_C0");
const end = html.indexOf("// Minimal 4x4 maths");
if (start < 0 || end < 0) {
  console.error("FAIL  could not locate the parser block in walkthrough.html");
  process.exit(1);
}
const source = html.slice(start, end).replace(/\/\/ -+\n?/g, "");
// new Function() on a string is a code-injection risk when the string comes
// from anywhere untrusted. Here the input is a file in this repository, read
// from a path derived from this script's own location, in a test that only ever
// runs locally — the alternative (duplicating the parser into a module) would
// let the copy drift from the code that actually ships in the page, which is
// the failure this test exists to prevent.
const parsePLY = new Function(source + "\nreturn parsePLY;")();

const fixtures = join(ROOT, "tools", "fixtures");
if (!existsSync(join(fixtures, "cloud.ply"))) {
  console.error("FAIL  fixtures missing - run: python3 tools/make_viewer_fixtures.py");
  process.exit(1);
}
const load = name => {
  const buf = readFileSync(join(fixtures, name));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
};

// --- plain coloured point cloud --------------------------------------------
const cloud = parsePLY(load("cloud.ply"));
check("point cloud vertex count", cloud.count === 1000);
check("positions array is sized correctly", cloud.positions.length === 3000);
check("colors array is sized correctly", cloud.colors.length === 3000);

// The fixture's first vertex is a known sentinel, so a stride error shows up
// immediately rather than as a vaguely wrong-looking cloud.
check("first position is exact",
  Math.abs(cloud.positions[0] - 1.5) < 1e-6 &&
  Math.abs(cloud.positions[1] + 2.25) < 1e-6 &&
  Math.abs(cloud.positions[2] - 3.75) < 1e-6);
check("first colour is exact",
  cloud.colors[0] === 255 && cloud.colors[1] === 128 && cloud.colors[2] === 7);

// Last vertex catches an off-by-one in the stride that the first would not.
const n = cloud.count - 1;
check("last position is exact",
  Math.abs(cloud.positions[n*3] - 9.5) < 1e-6 &&
  Math.abs(cloud.positions[n*3+2] + 0.25) < 1e-6);
check("last colour is exact",
  cloud.colors[n*3] === 3 && cloud.colors[n*3+1] === 200 && cloud.colors[n*3+2] === 99);

// --- 3DGS splat file --------------------------------------------------------
// Same geometry, but colour is stored as spherical-harmonic DC coefficients.
// Reading f_dc as raw RGB yields a washed-out grey scene; the SH constant must
// be undone. Fixture encodes pure red, so a correct decode returns red.
const splat = parsePLY(load("splat.ply"));
check("splat vertex count", splat.count === 1000);
check("splat positions match the point cloud",
  Math.abs(splat.positions[0] - cloud.positions[0]) < 1e-5 &&
  Math.abs(splat.positions[1] - cloud.positions[1]) < 1e-5);
check("SH degree-0 colour decodes to red, not grey",
  splat.colors[0] > 240 && splat.colors[1] < 20 && splat.colors[2] < 20);
check("SH decode handles a mid-grey correctly",
  Math.abs(splat.colors[3] - 128) <= 2 &&
  Math.abs(splat.colors[4] - 128) <= 2);

// --- ascii ------------------------------------------------------------------
const ascii = parsePLY(load("ascii.ply"));
check("ascii format parses", ascii.count === 3);
check("ascii positions are right", Math.abs(ascii.positions[3] - 4) < 1e-6);
check("ascii colours are right", ascii.colors[3] === 10 && ascii.colors[4] === 20);

// --- failure modes must be loud, not silent ---------------------------------
const expectThrow = (label, buf) => {
  let threw = false, msg = "";
  try { parsePLY(buf); } catch (e) { threw = true; msg = e.message; }
  check(label, threw);
  return msg;
};
expectThrow("rejects a non-PLY file",
  new TextEncoder().encode("this is not a ply file at all").buffer);

// A truncated file is the realistic corruption: an interrupted download. It
// must report that rather than rendering a partial cloud as if complete.
const full = new Uint8Array(load("cloud.ply"));
const cut = full.slice(0, full.length - 6000);
const msg = expectThrow("rejects a truncated file", cut.buffer);
check("truncation error explains itself", /truncated/i.test(msg));

// --- report -----------------------------------------------------------------
let failed = 0;
for (const [label, ok] of results) {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}`);
  if (!ok) failed++;
}
console.log(`\n${results.length - failed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
