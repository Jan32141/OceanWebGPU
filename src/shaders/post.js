import { constants, globals, fullscreenVS } from './common.js';

// Dual-filter bloom (13-tap downsample / 9-tap tent upsample, as in CoD: AW).
export const bloomDownWGSL = /* wgsl */ `
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var dst : texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var samp : sampler;

fn tap(uv : vec2f) -> vec3f {
  let c = textureSampleLevel(src, samp, uv, 0.0).rgb;
  // Guard against NaN/inf from extreme specular.
  return select(min(c, vec3f(60000.0)), vec3f(0.0), c != c);
}

fn karis(c : vec3f) -> f32 { return 1.0 / (1.0 + dot(c, vec3f(0.2126, 0.7152, 0.0722))); }

fn downsample(id : vec2u, firstPass : bool) -> vec3f {
  let size = vec2f(textureDimensions(dst));
  let uv = (vec2f(id) + 0.5) / size;
  let t = 1.0 / vec2f(textureDimensions(src));
  let a = tap(uv + t * vec2f(-2.0, -2.0)); let b = tap(uv + t * vec2f(0.0, -2.0)); let c = tap(uv + t * vec2f(2.0, -2.0));
  let d = tap(uv + t * vec2f(-2.0, 0.0));  let e = tap(uv);                         let f = tap(uv + t * vec2f(2.0, 0.0));
  let g = tap(uv + t * vec2f(-2.0, 2.0));  let h = tap(uv + t * vec2f(0.0, 2.0));   let i = tap(uv + t * vec2f(2.0, 2.0));
  let j = tap(uv + t * vec2f(-1.0, -1.0)); let k = tap(uv + t * vec2f(1.0, -1.0));
  let l = tap(uv + t * vec2f(-1.0, 1.0));  let m = tap(uv + t * vec2f(1.0, 1.0));
  if (!firstPass) {
    return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  }
  // Karis average on the first pass suppresses fireflies from sun glitter.
  let g0 = (a + b + d + e) * 0.25; let g1 = (b + c + e + f) * 0.25;
  let g2 = (d + e + g + h) * 0.25; let g3 = (e + f + h + i) * 0.25;
  let g4 = (j + k + l + m) * 0.25;
  let w0 = karis(g0) * 0.125; let w1 = karis(g1) * 0.125; let w2 = karis(g2) * 0.125;
  let w3 = karis(g3) * 0.125; let w4 = karis(g4) * 0.5;
  return (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
}

@compute @workgroup_size(8, 8)
fn first(@builtin(global_invocation_id) id : vec3u) {
  if (any(id.xy >= textureDimensions(dst))) { return; }
  textureStore(dst, id.xy, vec4f(downsample(id.xy, true), 1.0));
}

@compute @workgroup_size(8, 8)
fn down(@builtin(global_invocation_id) id : vec3u) {
  if (any(id.xy >= textureDimensions(dst))) { return; }
  textureStore(dst, id.xy, vec4f(downsample(id.xy, false), 1.0));
}
`;

export const bloomUpWGSL = /* wgsl */ `
@group(0) @binding(0) var small : texture_2d<f32>;
@group(0) @binding(1) var cur : texture_2d<f32>;
@group(0) @binding(2) var dst : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var samp : sampler;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let size = textureDimensions(dst);
  if (any(id.xy >= size)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
  let t = 1.0 / vec2f(size);
  var s = textureSampleLevel(small, samp, uv, 0.0).rgb * 4.0;
  s += (textureSampleLevel(small, samp, uv + t * vec2f(-1.0, 0.0), 0.0).rgb +
        textureSampleLevel(small, samp, uv + t * vec2f(1.0, 0.0), 0.0).rgb +
        textureSampleLevel(small, samp, uv + t * vec2f(0.0, -1.0), 0.0).rgb +
        textureSampleLevel(small, samp, uv + t * vec2f(0.0, 1.0), 0.0).rgb) * 2.0;
  s += textureSampleLevel(small, samp, uv + t * vec2f(-1.0, -1.0), 0.0).rgb +
       textureSampleLevel(small, samp, uv + t * vec2f(1.0, -1.0), 0.0).rgb +
       textureSampleLevel(small, samp, uv + t * vec2f(-1.0, 1.0), 0.0).rgb +
       textureSampleLevel(small, samp, uv + t * vec2f(1.0, 1.0), 0.0).rgb;
  let c = textureLoad(cur, vec2i(id.xy), 0).rgb + s / 16.0;
  textureStore(dst, id.xy, vec4f(c, 1.0));
}
`;

export const tonemapWGSL = /* wgsl */ `
${constants}
${globals}
@group(0) @binding(0) var<uniform> G : Globals;
@group(0) @binding(1) var hdr : texture_2d<f32>;
@group(0) @binding(2) var bloomTex : texture_2d<f32>;
@group(0) @binding(3) var samp : sampler;
@group(0) @binding(4) var<uniform> bloomLevels : vec4f;

${fullscreenVS}

// Stephen Hill's ACES fit.
fn aces(x : vec3f) -> vec3f {
  let inM = mat3x3f(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777);
  let outM = mat3x3f(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602);
  let v = inM * x;
  let a = v * (v + 0.0245786) - 0.000090537;
  let b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return saturate(outM * (a / b));
}

fn toSRGB(c : vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}

@fragment
fn fs(@builtin(position) fc : vec4f) -> @location(0) vec4f {
  let uv = fc.xy / G.resolution;
  var c = textureLoad(hdr, vec2i(fc.xy), 0).rgb;
  c = select(c, vec3f(0.0), c != c);
  let b = textureSampleLevel(bloomTex, samp, uv, 0.0).rgb / bloomLevels.x;
  c = mix(c, b, G.bloom);
  // Gentle vignette.
  let q = uv - 0.5;
  c *= 1.0 - dot(q, q) * 0.35;
  c = toSRGB(aces(c * G.exposure));
  // Dither to hide banding in the sky gradient.
  let n = fract(sin(dot(fc.xy, vec2f(12.9898, 78.233))) * 43758.5453);
  return vec4f(c + (n - 0.5) / 255.0, 1.0);
}
`;
