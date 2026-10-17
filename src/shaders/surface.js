import { constants, globals, noise, atmosphere, transmittance, skyView, clouds } from './common.js';
import { terrainFns } from './terrain.js';

export const surfaceWGSL = /* wgsl */ `
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
@group(0) @binding(5) var dispTex : texture_2d_array<f32>;
@group(0) @binding(6) var derivTex : texture_2d_array<f32>;
@group(0) @binding(7) var envTex : texture_2d_array<f32>;
@group(0) @binding(8) var repeatS : sampler;
@group(0) @binding(9) var heightTex : texture_2d<f32>;

${terrainFns}

const TEX_SIZE : f32 = 256.0;

// Waves lose height in shallow water; keep a little motion right up to the beach.
fn shoreAttenuation(depth : f32) -> f32 {
  return mix(0.35, 1.0, smoothstep(0.0, 8.0, depth));
}

struct VOut {
  @builtin(position) pos : vec4f,
  @location(0) rel : vec3f,     // displaced position relative to the camera
  @location(1) xz : vec2f,      // undisplaced world xz (Lagrangian coordinate)
  @location(2) height : f32,
};

@vertex
fn vs(@location(0) offset : vec2f, @location(1) spacing : f32) -> VOut {
  // The polar grid follows the camera; waves are sampled in world space.
  let xz = G.camPos.xz + offset;
  var d = vec3f(0.0);
  for (var c = 0u; c < CASCADES; c++) {
    let L = G.lengths[c];
    // Pick a mip whose texel matches the local vertex spacing (no aliasing far away).
    let lod = clamp(log2(spacing / (L / TEX_SIZE)), 0.0, 8.0);
    d += textureSampleLevel(dispTex, repeatS, xz / L, c, lod).xyz;
  }
  d *= shoreAttenuation(max(-terrainHeight(xz), 0.0));
  let world = vec3f(xz.x + d.x, d.y, xz.y + d.z);
  var o : VOut;
  o.rel = world - G.camPos;
  o.pos = G.viewProj * vec4f(o.rel, 1.0);
  o.xz = xz;
  o.height = d.y;
  return o;
}

fn ggx(nh : f32, a : f32) -> f32 {
  let a2 = a * a;
  let d = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

fn smithVis(nv : f32, nl : f32, a : f32) -> f32 {
  let a2 = a * a;
  let gv = nl * sqrt(nv * nv * (1.0 - a2) + a2);
  let gl = nv * sqrt(nl * nl * (1.0 - a2) + a2);
  return 0.5 / max(gv + gl, 1e-5);
}

fn fresnel(c : f32) -> f32 {
  return 0.02 + 0.98 * pow(1.0 - saturate(c), 5.0);
}

fn envUV(d : vec3f) -> vec2f {
  return vec2f(fract(atan2(d.z, d.x) / TAU + 1.0), acos(clamp(d.y, -1.0, 1.0)) / PI);
}

@fragment
fn fs(v : VOut) -> @location(0) vec4f {
  // Sum slopes and foam over the cascades (hardware mips + anisotropic filtering).
  var slope = vec2f(0.0);
  var foam = 0.0;
  for (var c = 0u; c < CASCADES; c++) {
    let uv = v.xz / G.lengths[c];
    let dv = textureSample(derivTex, repeatS, uv, c);
    let ds = textureSample(dispTex, repeatS, uv, c);
    slope += dv.xy / max(vec2f(1.0) + dv.zw, vec2f(0.2));
    foam += ds.w * (1.0 - 0.25 * f32(c));
  }
  let depth = max(-terrainHeight(v.xz), 0.0);
  slope *= shoreAttenuation(depth);
  // Pixel footprint in metres drives roughness for unresolved waves.
  let footprint = max(length(dpdx(v.xz)), length(dpdy(v.xz)));

  let dist = length(v.rel);
  let V = -v.rel / dist;
  var n = normalize(vec3f(-slope.x, 1.0, -slope.y));
  let nvRaw = dot(n, V);
  if (nvRaw < 0.02) { n = normalize(n + V * (0.02 - nvRaw)); }

  let L = G.sunDir;
  let alpha = clamp(0.035 + 0.035 * log2(1.0 + footprint * 3.0), 0.035, 0.28);
  let nl = max(dot(n, L), 0.0);
  let nv = max(dot(n, V), 1e-4);
  let H = normalize(L + V);
  let nh = max(dot(n, H), 0.0);
  let vh = max(dot(V, H), 0.0);

  let worldPos = v.rel + G.camPos;
  let sunE = G.sunIntensity * sampleTransmittance(RG + 1e-5, L.y) * cloudShadow(worldPos) *
             terrainShadow(vec3f(worldPos.x, max(worldPos.y, 0.0) + 0.3, worldPos.z), L);

  // Sky + cloud reflection from the environment map, blurred by roughness.
  var R = reflect(-V, n);
  R.y = abs(R.y);
  let lod = clamp(log2(max(2.0 * alpha / 0.0123, 1.0)), 0.0, 7.0);
  let refl = textureSampleLevel(envTex, wrapU, envUV(R), 0, lod).rgb;

  // Sun glitter: GGX specular on per-pixel normals.
  let spec = sunE * ggx(nh, alpha) * smithVis(nv, nl, alpha) * fresnel(vh) * nl;

  // Water body: deep scattering + subsurface glow through backlit crests.
  let amb = 0.5 * (textureSampleLevel(envTex, wrapU, vec2f(0.25, 0.25), 0, 7.0).rgb +
                   textureSampleLevel(envTex, wrapU, vec2f(0.75, 0.25), 0, 7.0).rgb);
  let deep = vec3f(0.0035, 0.016, 0.024);
  let sssColor = vec3f(0.06, 0.30, 0.27);
  let crest = max(v.height + 0.6, 0.0);
  let backlit = pow(saturate(dot(-V, L) * 0.5 + 0.5), 4.0) * pow(saturate(0.5 - 0.5 * dot(L, n)), 3.0);
  let sss = sunE * sssColor * (0.9 * crest * backlit + 0.015 * pow(nv, 2.0));
  var body = deep * (amb + sunE * nl * 0.05) + sss;
  // Shallow water: the sandy bottom shows through as turquoise.
  let shallow = vec3f(0.10, 0.30, 0.26) * (amb + sunE * max(L.y, 0.0) * 0.5);
  body = mix(shallow, body, smoothstep(0.0, 5.0, depth));

  let F = fresnel(nv);
  var col = (1.0 - F) * body + F * refl + spec;

  // Whitecaps: Jacobian-driven coverage broken up by procedural detail.
  let fineNoise = fbm(v.xz * 0.9 + vec2f(G.time * 0.05, 0.0), 4);
  let coarseNoise = fbm(v.xz * 0.15 - vec2f(0.0, G.time * 0.02), 3);
  let detail = mix(fineNoise * 0.6 + coarseNoise * 0.4, 0.5, saturate(footprint * 0.4));
  let foamMask = saturate(foam * G.foamAmount);
  var coverage = saturate((foamMask - (1.0 - detail)) * 3.0 + foamMask * 0.15);
  // Surf line: bands of foam rolling up the beach.
  let surf = (1.0 - smoothstep(0.0, 1.8, depth)) * step(1e-3, depth);
  let bands = 0.5 + 0.5 * sin(depth * 5.0 - G.time * 1.3 + fineNoise * 6.0);
  coverage = max(coverage, saturate(surf * (bands * 0.8 + 0.3) * (fineNoise * 1.6 - 0.1)));
  let foamLit = vec3f(0.85) * (sunE * (nl * 0.6 + 0.4) / PI + amb);
  col = mix(col, foamLit, coverage);

  // Aerial perspective toward the horizon.
  let haze = 1.0 - exp(-dist / 42000.0);
  col = mix(col, skyAbove(-V), haze);

  return vec4f(col, 1.0);
}
`;
