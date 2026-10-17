import { constants, FFT_SIZE } from './common.js';

const LOG_N = Math.log2(FFT_SIZE);

// Must match the Float32Array layout written in ocean.js (24 floats).
const simParams = /* wgsl */ `
${constants}
const N : u32 = ${FFT_SIZE}u;
const LOG_N : u32 = ${LOG_N}u;

struct Sim {
  lengths : vec4f,
  cutLow : vec4f,
  cutHigh : vec4f,
  windSpeed : f32,
  windDir : f32,
  fetch : f32,
  gravity : f32,
  depth : f32,
  time : f32,
  dt : f32,
  choppiness : f32,
  foamThreshold : f32,
  foamDecay : f32,
  spread : f32,
  seed : f32,
};

fn cmul(a : vec2f, b : vec2f) -> vec2f {
  return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
`;

const dispersionFn = /* wgsl */ `
fn dispersion(k : f32) -> f32 {
  return sqrt(P.gravity * k * tanh(min(k * P.depth, 20.0)));
}
`;

// Initial spectrum h0(k) and conj(h0(-k)) for every cascade.
// JONSWAP spectrum with Donelan-Banner directional spreading and finite depth dispersion.
export const initSpectrumWGSL = /* wgsl */ `
${simParams}
${dispersionFn}
@group(0) @binding(0) var<uniform> P : Sim;
@group(0) @binding(1) var h0Out : texture_storage_2d_array<rgba32float, write>;

fn pcg(v : u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn gauss2(x : u32, y : u32, c : u32) -> vec2f {
  let a = pcg(x + pcg(y + pcg(c + u32(P.seed) * 7919u)));
  let b = pcg(a ^ 0x9E3779B9u);
  let u = (vec2f(f32(a >> 8u), f32(b >> 8u)) + 0.5) / 16777216.0;
  let r = sqrt(-2.0 * log(u.x));
  return r * vec2f(cos(TAU * u.y), sin(TAU * u.y));
}

fn peakOmega() -> f32 {
  let U = max(P.windSpeed, 0.5);
  return 22.0 * pow(P.gravity * P.gravity / (U * P.fetch), 1.0 / 3.0);
}

fn jonswap(w : f32) -> f32 {
  let U = max(P.windSpeed, 0.5);
  let g = P.gravity;
  let alpha = 0.076 * pow(U * U / (P.fetch * g), 0.22);
  let wp = peakOmega();
  let sigma = select(0.09, 0.07, w <= wp);
  let r = exp(-(w - wp) * (w - wp) / (2.0 * sigma * sigma * wp * wp));
  return alpha * g * g / pow(w, 5.0) * exp(-1.25 * pow(wp / w, 4.0)) * pow(3.3, r);
}

fn donelanBanner(w : f32, theta : f32) -> f32 {
  let ratio = w / peakOmega();
  var beta : f32;
  if (ratio < 0.95) {
    beta = 2.61 * pow(ratio, 1.3);
  } else if (ratio < 1.6) {
    beta = 2.28 * pow(ratio, -1.3);
  } else {
    beta = pow(10.0, -0.4 + 0.8393 * exp(-0.567 * log(ratio * ratio)));
  }
  let s = 1.0 / cosh(beta * theta);
  return beta / (2.0 * tanh(beta * PI)) * s * s;
}

fn amplitude(k : vec2f, c : u32) -> f32 {
  let kl = length(k);
  if (kl < P.cutLow[c] || kl >= P.cutHigh[c]) { return 0.0; }
  let w = dispersion(kl);
  let kd = min(kl * P.depth, 20.0);
  let ch = cosh(kd);
  let dwdk = P.gravity * (tanh(kd) + kd / (ch * ch)) / (2.0 * w);
  var theta = atan2(k.y, k.x) - P.windDir;
  theta = theta - TAU * floor((theta + PI) / TAU);
  let D = mix(donelanBanner(w, theta), 1.0 / TAU, P.spread);
  // Damp capillary-scale waves a little to reduce shimmer.
  let fade = exp(-kl * kl * 1e-4);
  let S = jonswap(w) * D * dwdk / kl * fade;
  let dk = TAU / P.lengths[c];
  // E|h0|^2 = S dk^2 / 2 so that the height variance integrates to the spectrum.
  return 0.5 * sqrt(max(S, 0.0)) * dk;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let c = id.z;
  let n = vec2i(id.xy) - vec2i(i32(N / 2u));
  let k = vec2f(n) * TAU / P.lengths[c];
  let h0 = gauss2(id.x, id.y, c) * amplitude(k, c);
  let m = (vec2u(N) - id.xy) % vec2u(N);
  let hm = gauss2(m.x, m.y, c) * amplitude(-k, c);
  textureStore(h0Out, id.xy, c, vec4f(h0, hm.x, -hm.y));
}
`;

// Evolves the spectrum to time t and packs 8 real fields into 4 complex ones
// (two per rgba32f layer) so the inverse FFT yields them all at once.
export const timeSpectrumWGSL = /* wgsl */ `
${simParams}
${dispersionFn}
@group(0) @binding(0) var<uniform> P : Sim;
@group(0) @binding(1) var h0Tex : texture_2d_array<f32>;
@group(0) @binding(2) var fftOut : texture_storage_2d_array<rgba32float, write>;

// a + i*b for spectra of two real fields a and b.
fn pack(a : vec2f, b : vec2f) -> vec2f { return vec2f(a.x - b.y, a.y + b.x); }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let c = id.z;
  let h0 = textureLoad(h0Tex, vec2i(id.xy), i32(c), 0);
  let n = vec2i(id.xy) - vec2i(i32(N / 2u));
  let k = vec2f(n) * TAU / P.lengths[c];
  let kl = max(length(k), 1e-6);
  let ph = dispersion(kl) * P.time;
  // e^{-i w t} so waves travel along +k (with the wind).
  let e = vec2f(cos(ph), -sin(ph));
  let h = cmul(h0.xy, e) + cmul(h0.zw, vec2f(e.x, -e.y));
  let ih = vec2f(-h.y, h.x);

  let dx = ih * (k.x / kl);
  let dz = ih * (k.y / kl);
  let dydx = ih * k.x;
  let dydz = ih * k.y;
  let dxdx = -h * (k.x * k.x / kl);
  let dzdz = -h * (k.y * k.y / kl);
  let dxdz = -h * (k.x * k.y / kl);

  textureStore(fftOut, id.xy, 2u * c, vec4f(pack(dx, h), pack(dz, dydx)));
  textureStore(fftOut, id.xy, 2u * c + 1u, vec4f(pack(dydz, dxdx), pack(dzdz, dxdz)));
}
`;

// One workgroup transforms one full row (or column) in shared memory.
// Radix-2 decimation-in-time with bit-reversed loads; each texel holds two complex values.
export const fftWGSL = /* wgsl */ `
${simParams}
@group(0) @binding(0) var src : texture_2d_array<f32>;
@group(0) @binding(1) var dst : texture_storage_2d_array<rgba32float, write>;

var<workgroup> buf : array<array<vec4f, ${FFT_SIZE}>, 2>;

fn bitReverse(x : u32) -> u32 { return reverseBits(x) >> (32u - LOG_N); }

fn transform(i : u32) -> vec4f {
  var ping = 0u;
  for (var s = 1u; s <= LOG_N; s++) {
    let m = 1u << s;
    let hm = m >> 1u;
    let k = i & (m - 1u);
    let j = i - k;
    let kk = k & (hm - 1u);
    let angle = TAU * f32(kk) / f32(m);
    let tw = vec2f(cos(angle), sin(angle));
    let a = buf[ping][j + kk];
    let b = buf[ping][j + kk + hm];
    let tb = vec4f(cmul(tw, b.xy), cmul(tw, b.zw));
    buf[1u - ping][i] = select(a - tb, a + tb, k < hm);
    workgroupBarrier();
    ping = 1u - ping;
  }
  return buf[ping][i];
}

@compute @workgroup_size(${FFT_SIZE})
fn horizontal(@builtin(workgroup_id) wg : vec3u, @builtin(local_invocation_id) lid : vec3u) {
  let x = lid.x;
  let y = wg.x;
  let layer = wg.y;
  buf[0][bitReverse(x)] = textureLoad(src, vec2u(x, y), layer, 0);
  workgroupBarrier();
  textureStore(dst, vec2u(x, y), layer, transform(x));
}

@compute @workgroup_size(${FFT_SIZE})
fn vertical(@builtin(workgroup_id) wg : vec3u, @builtin(local_invocation_id) lid : vec3u) {
  let x = wg.x;
  let y = lid.x;
  let layer = wg.y;
  buf[0][bitReverse(y)] = textureLoad(src, vec2u(x, y), layer, 0);
  workgroupBarrier();
  textureStore(dst, vec2u(x, y), layer, transform(y));
}
`;

// Unpacks the FFT results into displacement + slope maps and accumulates foam
// from the Jacobian of the horizontal displacement (folding = breaking waves).
export const assembleWGSL = /* wgsl */ `
${simParams}
@group(0) @binding(0) var<uniform> P : Sim;
@group(0) @binding(1) var fftTex : texture_2d_array<f32>;
@group(0) @binding(2) var prevDisp : texture_2d_array<f32>;
@group(0) @binding(3) var dispOut : texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var derivOut : texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let c = id.z;
  let p = vec2i(id.xy);
  // The spectrum is centred at N/2, which shifts the output by (-1)^(x+y).
  let sgn = select(-1.0, 1.0, ((id.x + id.y) & 1u) == 0u);
  let a = textureLoad(fftTex, p, i32(2u * c), 0) * sgn;
  let b = textureLoad(fftTex, p, i32(2u * c + 1u), 0) * sgn;

  let lam = P.choppiness;
  let disp = vec3f(lam * a.x, a.y, lam * a.z);
  let dxdx = lam * b.y;
  let dzdz = lam * b.z;
  let dxdz = lam * b.w;
  let jacobian = (1.0 + dxdx) * (1.0 + dzdz) - dxdz * dxdz;

  let prev = textureLoad(prevDisp, p, i32(c), 0).w;
  let inject = saturate((P.foamThreshold - jacobian) * 1.5);
  let foam = max(prev * exp(-P.foamDecay * P.dt), inject);

  textureStore(dispOut, id.xy, c, vec4f(disp, foam));
  textureStore(derivOut, id.xy, c, vec4f(a.w, b.x, dxdx, dzdz));
}
`;

// 2x2 box downsample for rgba16f texture arrays.
export const mipWGSL = /* wgsl */ `
@group(0) @binding(0) var src : texture_2d_array<f32>;
@group(0) @binding(1) var dst : texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let p = vec2i(id.xy) * 2;
  let l = i32(id.z);
  let c = textureLoad(src, p, l, 0) + textureLoad(src, p + vec2i(1, 0), l, 0) +
          textureLoad(src, p + vec2i(0, 1), l, 0) + textureLoad(src, p + vec2i(1, 1), l, 0);
  textureStore(dst, id.xy, id.z, c * 0.25);
}
`;
