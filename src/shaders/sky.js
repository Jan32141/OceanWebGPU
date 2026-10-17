import {
  constants,
  globals,
  noise,
  atmosphere,
  transmittance,
  skyViewMapping,
  skyView,
  clouds,
  fullscreenVS,
} from './common.js';

export const transmittanceWGSL = /* wgsl */ `
${constants}
${atmosphere}
@group(0) @binding(0) var outLUT : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let size = textureDimensions(outLUT);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
  // Invert the Bruneton mapping.
  let H = sqrt(RT * RT - RG * RG);
  let rho = H * uv.y;
  let r = sqrt(rho * rho + RG * RG);
  let dMin = RT - r;
  let dMax = rho + H;
  let d = dMin + uv.x * (dMax - dMin);
  var mu = 1.0;
  if (d > 0.0) { mu = clamp((H * H - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0); }

  let ro = vec3f(0.0, r, 0.0);
  let rd = vec3f(sqrt(1.0 - mu * mu), mu, 0.0);
  let steps = 64;
  let dt = d / f32(steps);
  var depth = vec3f(0.0);
  for (var i = 0; i < steps; i++) {
    let p = ro + rd * ((f32(i) + 0.5) * dt);
    depth += extinctionAt(length(p) - RG) * dt;
  }
  textureStore(outLUT, id.xy, vec4f(exp(-depth), 1.0));
}
`;

// Single scattering (Rayleigh + Mie + ozone absorption) toward every direction,
// as seen from the camera altitude. Re-rendered each frame (sun can move).
export const skyViewWGSL = /* wgsl */ `
${constants}
${globals}
${atmosphere}
${transmittance}
@group(0) @binding(0) var<uniform> G : Globals;
@group(0) @binding(1) var transLUT : texture_2d<f32>;
@group(0) @binding(2) var linClamp : sampler;
@group(0) @binding(3) var outLUT : texture_storage_2d<rgba16float, write>;

${skyViewMapping}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let size = textureDimensions(outLUT);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let rd = skyViewDir((vec2f(id.xy) + 0.5) / vec2f(size));
  let ro = vec3f(0.0, RG + max(G.camPos.y * 0.001, 0.0005), 0.0);

  let top = raySphere(ro, rd, RT);
  var tMax = top.y;
  let ground = raySphere(ro, rd, RG);
  if (ground.x > 0.0) { tMax = min(tMax, ground.x); }

  let cosT = dot(rd, G.sunDir);
  let phR = rayleighPhase(cosT);
  let phM = miePhase(cosT, MIE_G);
  var T = vec3f(1.0);
  var L = vec3f(0.0);
  let steps = 40;
  for (var i = 0; i < steps; i++) {
    // Quadratic step distribution: dense near the viewer.
    let t0 = tMax * pow(f32(i) / f32(steps), 2.0);
    let t1 = tMax * pow(f32(i + 1) / f32(steps), 2.0);
    let dt = t1 - t0;
    let p = ro + rd * (0.5 * (t0 + t1));
    let r = length(p);
    let h = r - RG;
    let ext = extinctionAt(h);
    let sR = RAY_S * densityRayleigh(h);
    let sM = MIE_S * densityMie(h);
    let sunT = sampleTransmittance(r, dot(p / r, G.sunDir));
    // Single scattering plus a crude isotropic term standing in for multiple scattering.
    let scatter = (sR * phR + sM * phM) * sunT + (sR + sM) * sunT * (0.25 / (4.0 * PI));
    let S = scatter * G.sunIntensity;
    let stepT = exp(-ext * dt);
    L += T * (S - S * stepT) / max(ext, vec3f(1e-7));
    T *= stepT;
  }
  textureStore(outLUT, id.xy, vec4f(L, 1.0));
}
`;

const skyBindings = /* wgsl */ `
@group(0) @binding(0) var<uniform> G : Globals;
@group(0) @binding(1) var transLUT : texture_2d<f32>;
@group(0) @binding(2) var skyLUT : texture_2d<f32>;
@group(0) @binding(3) var linClamp : sampler;
@group(0) @binding(4) var wrapU : sampler;
`;

const skyCommon = /* wgsl */ `
${constants}
${globals}
${noise}
${atmosphere}
${transmittance}
${skyView}
${clouds}
${skyBindings}
`;

// Lat-long environment map (sky + clouds, no sun disk) for ocean reflections.
export const envWGSL = /* wgsl */ `
${skyCommon}
@group(0) @binding(5) var outEnv : texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let size = textureDimensions(outEnv);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
  let az = uv.x * TAU;
  let el = 0.5 * PI - uv.y * PI;
  let rd = vec3f(cos(el) * cos(az), sin(el), cos(el) * sin(az));
  var col = skyAbove(rd);
  if (rd.y > 0.0) {
    col = renderClouds(rd, col, 4);
  } else {
    col *= 0.5;
  }
  textureStore(outEnv, id.xy, 0u, vec4f(col, 1.0));
}
`;

// Full-resolution background sky: atmosphere + sun disk + clouds.
export const backgroundWGSL = /* wgsl */ `
${skyCommon}
${fullscreenVS}

@fragment
fn fs(@builtin(position) fc : vec4f) -> @location(0) vec4f {
  let ndc = vec2f(fc.x / G.resolution.x * 2.0 - 1.0, 1.0 - fc.y / G.resolution.y * 2.0);
  let p = G.invViewProj * vec4f(ndc, 1.0, 1.0);
  let rd = normalize(p.xyz / p.w);
  var col = skyAbove(rd);
  if (rd.y > 0.0) {
    col += sunDisk(rd);
    col = renderClouds(rd, col, 6);
  } else {
    col *= 0.5;
  }
  return vec4f(col, 1.0);
}
`;
