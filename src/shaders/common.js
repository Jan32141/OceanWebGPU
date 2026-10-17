// WGSL snippets shared between passes. Snippets that touch resources expect the
// including shader to declare them under the names used here:
//   G (Globals uniform), transLUT, skyLUT, linClamp, wrapU.

export const CASCADES = 3;
export const FFT_SIZE = 256;

export const constants = /* wgsl */ `
const PI : f32 = 3.14159265359;
const TAU : f32 = 6.28318530718;
const CASCADES : u32 = ${CASCADES}u;
`;

// Must match the Float32Array layout written in main.js (56 floats, 224 bytes).
export const globals = /* wgsl */ `
struct Globals {
  viewProj : mat4x4f,
  invViewProj : mat4x4f,
  camPos : vec3f,
  time : f32,
  sunDir : vec3f,
  sunIntensity : f32,
  lengths : vec4f,
  resolution : vec2f,
  exposure : f32,
  bloom : f32,
  cloudCoverage : f32,
  cloudDensity : f32,
  cloudOffset : vec2f,
  foamAmount : f32,
  cirrus : f32,
  pad0 : f32,
  pad1 : f32,
};
`;

export const noise = /* wgsl */ `
fn hash12(p : vec2f) -> f32 {
  var p3 = fract(vec3f(p.x, p.y, p.x) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn vnoise(p : vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash12(i);
  let b = hash12(i + vec2f(1.0, 0.0));
  let c = hash12(i + vec2f(0.0, 1.0));
  let d = hash12(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Normalised to roughly [0, 1] regardless of octave count.
fn fbm(p0 : vec2f, octaves : i32) -> f32 {
  var p = p0;
  var a = 0.5;
  var s = 0.0;
  var norm = 0.0;
  let m = mat2x2f(0.8, 0.6, -0.6, 0.8);
  for (var i = 0; i < octaves; i++) {
    s += a * vnoise(p);
    norm += a;
    p = m * p * 2.03 + vec2f(1.7, 9.2);
    a *= 0.5;
  }
  return s / norm;
}
`;

// Earth atmosphere (units: km), parameters from Hillaire 2020 / Bruneton 2017.
export const atmosphere = /* wgsl */ `
const RG : f32 = 6360.0;
const RT : f32 = 6460.0;
const RAY_S : vec3f = vec3f(5.802e-3, 13.558e-3, 33.1e-3);
const MIE_S : f32 = 3.996e-3;
const MIE_E : f32 = 4.440e-3;
const OZONE_A : vec3f = vec3f(0.650e-3, 1.881e-3, 0.085e-3);
const MIE_G : f32 = 0.8;

fn raySphere(ro : vec3f, rd : vec3f, r : f32) -> vec2f {
  let b = dot(ro, rd);
  let c = dot(ro, ro) - r * r;
  let d = b * b - c;
  if (d < 0.0) { return vec2f(-1.0); }
  let s = sqrt(d);
  return vec2f(-b - s, -b + s);
}

fn densityRayleigh(h : f32) -> f32 { return exp(-h / 8.0); }
fn densityMie(h : f32) -> f32 { return exp(-h / 1.2); }
fn densityOzone(h : f32) -> f32 { return max(0.0, 1.0 - abs(h - 25.0) / 15.0); }

fn extinctionAt(h : f32) -> vec3f {
  return RAY_S * densityRayleigh(h) + vec3f(MIE_E * densityMie(h)) + OZONE_A * densityOzone(h);
}

fn rayleighPhase(c : f32) -> f32 { return 3.0 / (16.0 * PI) * (1.0 + c * c); }

fn miePhase(c : f32, g : f32) -> f32 {
  let g2 = g * g;
  return 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + c * c)) / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * c, 1.5));
}

fn hgPhase(c : f32, g : f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

// Bruneton's transmittance LUT parameterisation.
fn transmittanceUV(r : f32, mu : f32) -> vec2f {
  let H = sqrt(RT * RT - RG * RG);
  let rho = sqrt(max((r - RG) * (r + RG), 0.0));
  let disc = r * r * (mu * mu - 1.0) + RT * RT;
  let d = max(-r * mu + sqrt(max(disc, 0.0)), 0.0);
  let dMin = RT - r;
  let dMax = rho + H;
  return vec2f((d - dMin) / (dMax - dMin), rho / H);
}
`;

export const transmittance = /* wgsl */ `
fn sampleTransmittance(r : f32, mu : f32) -> vec3f {
  let horizon = -sqrt(max((r - RG) * (r + RG), 0.0)) / r;
  let uv = transmittanceUV(r, max(mu, horizon + 1e-4));
  let t = textureSampleLevel(transLUT, linClamp, uv, 0.0).rgb;
  // Soft cut so the sun fades out as it sinks below the horizon instead of popping.
  return t * smoothstep(horizon - 0.012, horizon + 0.002, mu);
}
`;

export const skyViewMapping = /* wgsl */ `
fn skyViewUV(dir : vec3f) -> vec2f {
  let u = fract(atan2(dir.z, dir.x) / TAU + 1.0);
  let l = asin(clamp(dir.y, -1.0, 1.0)) / (0.5 * PI);
  // Non-linear latitude mapping concentrates texels near the horizon.
  let v = 0.5 - 0.5 * sign(l) * sqrt(abs(l));
  return vec2f(u, v);
}

fn skyViewDir(uv : vec2f) -> vec3f {
  let l = (0.5 - uv.y) * 2.0;
  let el = sign(l) * l * l * 0.5 * PI;
  let az = uv.x * TAU;
  return vec3f(cos(el) * cos(az), sin(el), cos(el) * sin(az));
}
`;

export const skyView = /* wgsl */ `
${skyViewMapping}

fn sampleSkyView(dir : vec3f) -> vec3f {
  return textureSampleLevel(skyLUT, wrapU, skyViewUV(dir), 0.0).rgb;
}

// Sky radiance with the below-horizon part replaced by the horizon colour,
// which is what distant ocean and reflections should see.
fn skyAbove(dir : vec3f) -> vec3f {
  let h = vec3f(dir.x, max(dir.y, 0.0015), dir.z);
  return sampleSkyView(normalize(h + vec3f(1e-6, 0.0, 0.0)));
}

fn sunDisk(rd : vec3f) -> vec3f {
  let cosA = dot(rd, G.sunDir);
  if (cosA <= 0.0) { return vec3f(0.0); }
  let ang = asin(min(length(cross(rd, G.sunDir)), 1.0));
  let radius = 0.0065;
  let edge = 1.0 - smoothstep(radius * 0.92, radius * 1.08, ang);
  if (edge <= 0.0) { return vec3f(0.0); }
  let x = min(ang / radius, 1.0);
  let mu = sqrt(1.0 - x * x);
  let limb = pow(vec3f(mu), vec3f(0.4, 0.5, 0.6));
  let camR = RG + max(G.camPos.y, 0.0) * 0.001;
  return G.sunIntensity * 800.0 * limb * edge * sampleTransmittance(camR, rd.y);
}
`;

export const clouds = /* wgsl */ `
const CLOUD_H : f32 = 1.5;      // km, cumulus layer altitude
const CLOUD_THICK : f32 = 0.7;  // km
const CIRRUS_H : f32 = 8.0;

fn cloudDensity(xz : vec2f, octaves : i32) -> f32 {
  let p = xz * 0.55 + G.cloudOffset;
  let w = vec2f(fbm(p * 0.4 + vec2f(3.1, 1.7), 3), fbm(p * 0.4 + vec2f(8.3, 2.9), 3)) - 0.5;
  let n = fbm(p + w * 1.8, octaves);
  let large = fbm(p * 0.11 + vec2f(5.2, 1.3), 2);
  let cov = saturate(G.cloudCoverage + (large - 0.5) * 0.7);
  let thr = mix(0.82, 0.28, cov);
  return saturate((n - thr) / max(1.0 - thr, 1e-3) * 2.5);
}

fn cirrusDensity(xz : vec2f, octaves : i32) -> f32 {
  let r = mat2x2f(0.8, 0.6, -0.6, 0.8);
  var p = (r * xz) * vec2f(0.035, 0.15) + G.cloudOffset * 0.35 + vec2f(13.0, 7.0);
  p.x += fbm(p * vec2f(1.0, 0.5) + 2.0, 3) * 1.4;
  let n = fbm(p, octaves);
  return saturate((n - 0.5) * 3.0) * G.cirrus;
}

fn sunAt(p : vec3f) -> vec3f {
  let r = length(p);
  return G.sunIntensity * sampleTransmittance(r, dot(p / r, G.sunDir));
}

fn cloudAmbient() -> vec3f {
  let s = normalize(vec3f(G.sunDir.x + 1e-5, 0.0, G.sunDir.z));
  let up = sampleSkyView(vec3f(0.0, 1.0, 0.0));
  let h1 = sampleSkyView(normalize(vec3f(s.x, 0.15, s.z)));
  let h2 = sampleSkyView(normalize(vec3f(-s.x, 0.15, -s.z)));
  return up * 0.5 + (h1 + h2) * 0.25;
}

fn renderClouds(rd : vec3f, skyCol : vec3f, octaves : i32) -> vec3f {
  var col = skyCol;
  if (rd.y <= 0.0) { return col; }
  let ro = vec3f(0.0, RG + max(G.camPos.y, 0.0) * 0.001, 0.0);
  let camXZ = G.camPos.xz * 0.001;
  let cosT = dot(rd, G.sunDir);
  let amb = cloudAmbient();

  // High, thin cirrus: single-scattering with a strong forward lobe.
  if (G.cirrus > 0.001) {
    let t = raySphere(ro, rd, RG + CIRRUS_H).y;
    let p = ro + rd * t;
    let d = cirrusDensity(camXZ + rd.xz * t, octaves);
    if (d > 0.0) {
      let phase = mix(hgPhase(cosT, 0.6), hgPhase(cosT, -0.1), 0.4) * 4.0 * PI;
      var c = sunAt(p) * phase * 0.1 + amb * 0.7;
      let fog = exp(-t * 0.01);
      c = c * fog + skyCol * (1.0 - fog);
      let a = (1.0 - exp(-d * 1.4)) * smoothstep(450.0, 150.0, t);
      col = mix(col, c, a);
    }
  }

  // Cumulus layer: 2.5D slab with a short light march toward the sun.
  let t = raySphere(ro, rd, RG + CLOUD_H).y;
  if (t > 0.0 && t < 180.0) {
    let p = ro + rd * t;
    let xz = camXZ + rd.xz * t;
    let d = cloudDensity(xz, octaves);
    if (d > 0.002) {
      let sigma = 18.0 * G.cloudDensity;
      let sy = max(G.sunDir.y, 0.015);
      let sl = length(G.sunDir.xz);
      let dirL = select(vec2f(1.0, 0.0), G.sunDir.xz / max(sl, 1e-4), sl > 1e-4);
      let path = min(CLOUD_THICK * 0.5 / sy, 4.0);
      var tau = d * sigma * CLOUD_THICK * 0.2;
      for (var i = 0; i < 4; i++) {
        let s = (f32(i) + 0.5) / 4.0 * path;
        tau += cloudDensity(xz + dirL * s * sl, max(octaves - 2, 2)) * sigma * path * 0.25;
      }
      // Cheap multiple-scattering approximation (sum of attenuated octaves).
      let ms = exp(-tau) + 0.5 * exp(-tau * 0.25) + 0.25 * exp(-tau * 0.0625);
      let phase = mix(hgPhase(cosT, 0.7), hgPhase(cosT, -0.2), 0.35) * 4.0 * PI;
      var c = sunAt(p) * ms * phase * 0.16 + amb * (1.0 - 0.45 * d);
      let fog = exp(-t * 0.018);
      c = c * fog + skyCol * (1.0 - fog);
      let thick = min(CLOUD_THICK * d / max(rd.y, 0.035), 8.0);
      let a = (1.0 - exp(-sigma * d * thick * 0.5)) * smoothstep(180.0, 70.0, t);
      col = mix(col, c, a);
    }
  }
  return col;
}

// Fraction of sunlight reaching a sea-level point (metres) through the cumulus layer.
fn cloudShadow(worldM : vec3f) -> f32 {
  let sy = max(G.sunDir.y, 0.04);
  let xz = worldM.xz * 0.001 + G.sunDir.xz / sy * CLOUD_H;
  let d = cloudDensity(xz, 3);
  return mix(1.0, exp(-d * 18.0 * G.cloudDensity * CLOUD_THICK * 0.35), 0.92);
}
`;

export const fullscreenVS = /* wgsl */ `
@vertex
fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  return vec4f(p, 0.0, 1.0);
}
`;
