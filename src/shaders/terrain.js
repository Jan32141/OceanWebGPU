import { constants, globals, noise, atmosphere, transmittance, skyView, clouds } from './common.js';

// Heightmap layout; must match terrain.js.
export const TERRAIN_RES = 1024;
export const TERRAIN_TEXEL = 4; // metres per texel
export const TERRAIN_SIZE = TERRAIN_RES * TERRAIN_TEXEL;
export const TERRAIN_MAX_HEIGHT = 200;
export const TILE = 64; // quads per tile edge

// Expects `heightTex : texture_2d<f32>` (r32float, sampled with textureLoad).
export const terrainFns = /* wgsl */ `
const TERRAIN_RES : i32 = ${TERRAIN_RES};
const TERRAIN_TEXEL : f32 = ${TERRAIN_TEXEL}.0;
const TERRAIN_HALF : f32 = ${TERRAIN_SIZE / 2}.0;
const TERRAIN_MAX_H : f32 = ${TERRAIN_MAX_HEIGHT}.0;

fn terrainLoad(p : vec2i) -> f32 {
  return textureLoad(heightTex, clamp(p, vec2i(0), vec2i(TERRAIN_RES - 1)), 0).r;
}

// Bilinear height at world xz (metres).
fn terrainHeight(xz : vec2f) -> f32 {
  let uv = (xz + TERRAIN_HALF) / TERRAIN_TEXEL - 0.5;
  let i = vec2i(floor(uv));
  let f = fract(uv);
  let a = terrainLoad(i);
  let b = terrainLoad(i + vec2i(1, 0));
  let c = terrainLoad(i + vec2i(0, 1));
  let d = terrainLoad(i + vec2i(1, 1));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Soft shadow from marching the heightfield toward the sun.
fn terrainShadow(p : vec3f, L : vec3f) -> f32 {
  if (L.y < -0.02) { return 0.0; }
  var t = 2.0;
  var res = 1.0;
  for (var i = 0; i < 28; i++) {
    let q = p + L * t;
    if (q.y > TERRAIN_MAX_H || abs(q.x) > TERRAIN_HALF || abs(q.z) > TERRAIN_HALF) { break; }
    res = min(res, 8.0 * (q.y - terrainHeight(q.xz)) / t);
    if (res < 0.0) { break; }
    t = t * 1.2 + 1.0;
  }
  return smoothstep(0.0, 1.0, res);
}
`;

export const terrainWGSL = /* wgsl */ `
${constants}
${globals}
${noise}
${atmosphere}
${transmittance}
${skyView}
${clouds}

@group(0) @binding(0) var<uniform> G : Globals;
@group(0) @binding(1) var transLUT : texture_2d<f32>;
@group(0) @binding(2) var skyLUT : texture_2d<f32>;
@group(0) @binding(3) var linClamp : sampler;
@group(0) @binding(4) var wrapU : sampler;
@group(0) @binding(5) var heightTex : texture_2d<f32>;

${terrainFns}

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) rel : vec3f,
  @location(1) normal : vec3f,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32, @location(0) tile : vec2f) -> VOut {
  let p = vec2i(tile) + vec2i(i32(vi % ${TILE + 1}u), i32(vi / ${TILE + 1}u));
  let h = terrainLoad(p);
  let xz = (vec2f(p) + 0.5) * TERRAIN_TEXEL - TERRAIN_HALF;
  let hl = terrainLoad(p - vec2i(1, 0));
  let hr = terrainLoad(p + vec2i(1, 0));
  let hd = terrainLoad(p - vec2i(0, 1));
  let hu = terrainLoad(p + vec2i(0, 1));
  var o : VOut;
  o.rel = vec3f(xz.x, h, xz.y) - G.camPos;
  o.pos = G.viewProj * vec4f(o.rel, 1.0);
  o.normal = normalize(vec3f(hl - hr, 2.0 * TERRAIN_TEXEL, hd - hu));
  return o;
}

@fragment
fn fs(v : VOut) -> @location(0) vec4f {
  let world = v.rel + G.camPos;
  let xz = world.xz;
  let h = world.y;

  // Detail normals. Kept subtle: with a grazing sunset sun, broad bumps read as dark blotches.
  let dist = length(v.rel);
  let e = 0.6;
  let n0 = fbm(xz * 0.35, 3);
  let nx = fbm((xz + vec2f(e, 0.0)) * 0.35, 3);
  let nz = fbm((xz + vec2f(0.0, e)) * 0.35, 3);
  var n = v.normal + vec3f(n0 - nx, 0.0, n0 - nz) * 0.3;
  // Wind ripples in the sand, faded out with distance to avoid aliasing.
  let rippleFade = (1.0 - smoothstep(8.0, 45.0, dist)) * (1.0 - smoothstep(2.5, 5.0, h));
  let rp = dot(xz, vec2f(0.8, 0.6)) * 7.0 + vnoise(xz * 0.8) * 5.0;
  n += vec3f(0.8, 0.0, 0.6) * cos(rp) * 0.07 * rippleFade;
  n = normalize(n);
  let slope = 1.0 - n.y;

  let broad = fbm(xz * 0.02, 3);
  let fine = vnoise(xz * 1.3);
  let sandLine = 2.0 + broad * 3.0;

  var sand = vec3f(0.50, 0.42, 0.29) * (0.85 + 0.3 * fine);
  sand *= mix(1.0, 0.5, smoothstep(0.9, 0.1, h)); // wet sand at the waterline
  let grass = mix(vec3f(0.045, 0.085, 0.022), vec3f(0.14, 0.12, 0.05), broad) * (0.8 + 0.4 * fine);
  let rock = vec3f(0.19, 0.17, 0.15) * (0.75 + 0.5 * fine);

  var albedo = sand;
  let grassAmt = smoothstep(sandLine, sandLine + 1.5, h) * (1.0 - smoothstep(0.25, 0.4, slope));
  albedo = mix(albedo, grass, grassAmt);
  let rockAmt = smoothstep(0.38, 0.55, slope + (fine - 0.5) * 0.12) * smoothstep(1.0, 3.0, h);
  albedo = mix(albedo, rock, rockAmt);

  let L = G.sunDir;
  let nl = max(dot(n, L), 0.0);
  let shadow = terrainShadow(world + v.normal * 0.5, L) * cloudShadow(world);
  let sunE = G.sunIntensity * sampleTransmittance(RG + max(h, 0.0) * 0.001 + 1e-5, L.y) * shadow;
  let amb = cloudAmbient();
  var col = albedo / PI * sunE * nl + albedo * amb * (0.55 + 0.45 * n.y);
  // Seabed seen through shallow water: darker and tinted by absorption.
  col *= mix(vec3f(1.0), vec3f(0.45, 0.72, 0.68), smoothstep(0.0, -2.5, h));

  let haze = 1.0 - exp(-dist / 42000.0);
  col = mix(col, skyAbove(-v.rel / dist), haze);
  return vec4f(col, 1.0);
}
`;
