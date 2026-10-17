import { terrainWGSL, TERRAIN_RES, TERRAIN_TEXEL, TERRAIN_SIZE, TILE } from './shaders/terrain.js';
import { texture, bindGroup, shaderModule } from './gpu.js';
import { skyLayoutEntries, HDR_FORMAT, DEPTH_FORMAT } from './sky.js';

// Island centres (m), radii and peak heights. The main island sits behind the
// start position so its beach faces the default sunset (-Z).
const ISLANDS = [
  { x: 0, z: 620, r: 820, h: 95 },
  { x: -1250, z: -350, r: 480, h: 55 },
  { x: 1250, z: 150, r: 600, h: 125 },
  { x: 420, z: -1350, r: 340, h: 30 },
  { x: -750, z: 1500, r: 440, h: 70 },
];
const SEABED = -38;

function hash(ix, iz) {
  let h = Math.imul(ix, 374761393) + Math.imul(iz, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function vnoise(x, z) {
  const ix = Math.floor(x);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uz = fz * fz * (3 - 2 * fz);
  const a = hash(ix, iz);
  const b = hash(ix + 1, iz);
  const c = hash(ix, iz + 1);
  const d = hash(ix + 1, iz + 1);
  return a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz;
}

function fbm(x, z, octaves, ridged = false) {
  let s = 0;
  let a = 0.5;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    let n = vnoise(x, z);
    if (ridged) n = (1 - Math.abs(2 * n - 1)) ** 2;
    s += a * n;
    norm += a;
    const nx = 0.8 * x - 0.6 * z;
    z = (0.6 * x + 0.8 * z) * 2.03 + 9.2;
    x = nx * 2.03 + 1.7;
    a *= 0.5;
  }
  return s / norm;
}

function islandHeight(x, z) {
  const warp = fbm(x * 0.0035 + 11.3, z * 0.0035 + 4.1, 4);
  let s = 0;
  let peak = 0;
  for (const is of ISLANDS) {
    const d = Math.hypot(x - is.x, z - is.z) / (is.r * (0.7 + 0.6 * warp));
    const m = Math.max(0, 1 - d);
    const sm = m * m * (3 - 2 * m);
    if (sm > s) {
      s = sm;
      peak = is.h;
    }
  }
  if (s <= 0) return SEABED;
  const ridge = fbm(x * 0.006 + 3.7, z * 0.006 - 1.9, 5, true);
  let e = SEABED + s * 46 + Math.pow(s, 2.2) * peak * (0.45 + 0.75 * ridge);
  e += (fbm(x * 0.03, z * 0.03, 3) - 0.5) * 3 * Math.min(1, s * 3);
  // Flatten the band around sea level into beaches and shallow sand flats.
  // Gentler near sea level gives sandy beaches, but keep enough slope for a clean shoreline.
  if (e > 0 && e < 8) e = 8 * (e / 8) ** 1.5;
  else if (e <= 0 && e > -10) e = -10 * (-e / 10) ** 1.25;
  return e;
}

export class Terrain {
  constructor(device, sky) {
    const res = TERRAIN_RES;
    const heights = new Float32Array(res * res);
    for (let j = 0; j < res; j++) {
      const z = (j + 0.5) * TERRAIN_TEXEL - TERRAIN_SIZE / 2;
      for (let i = 0; i < res; i++) {
        heights[j * res + i] = islandHeight((i + 0.5) * TERRAIN_TEXEL - TERRAIN_SIZE / 2, z);
      }
    }
    this.heights = heights;

    this.heightTex = device.createTexture({
      label: 'terrain height',
      size: [res, res],
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: this.heightTex }, heights, { bytesPerRow: res * 4 }, [res, res]);
    this.heightView = this.heightTex.createView();

    // Only draw tiles that reach near the surface; deeper seabed is hidden by the opaque ocean.
    const tiles = [];
    const n = res / TILE;
    for (let ty = 0; ty < n; ty++) {
      for (let tx = 0; tx < n; tx++) {
        let max = -Infinity;
        for (let j = ty * TILE; j <= Math.min(res - 1, (ty + 1) * TILE); j++) {
          for (let i = tx * TILE; i <= Math.min(res - 1, (tx + 1) * TILE); i++) max = Math.max(max, heights[j * res + i]);
        }
        if (max > -8) tiles.push(tx * TILE, ty * TILE);
      }
    }
    this.tileCount = tiles.length / 2;
    this.tileBuffer = device.createBuffer({
      label: 'terrain tiles',
      size: Math.max(8, tiles.length * 4),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.tileBuffer, 0, new Float32Array(tiles));

    const v = TILE + 1;
    const indices = new Uint16Array(TILE * TILE * 6);
    let o = 0;
    for (let y = 0; y < TILE; y++) {
      for (let x = 0; x < TILE; x++) {
        const a = y * v + x;
        indices.set([a, a + v, a + 1, a + 1, a + v, a + v + 1], o);
        o += 6;
      }
    }
    this.indexBuffer = device.createBuffer({
      label: 'terrain indices',
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.indexBuffer, 0, indices);
    this.indexCount = indices.length;

    const layout = device.createBindGroupLayout({
      label: 'terrain',
      entries: [...skyLayoutEntries(), texture(5, { type: 'unfilterable-float' })],
    });
    const module = shaderModule(device, 'terrain', terrainWGSL);
    this.pipeline = device.createRenderPipeline({
      label: 'terrain',
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: {
        module,
        entryPoint: 'vs',
        buffers: [{ arrayStride: 8, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] }],
      },
      fragment: { module, entryPoint: 'fs', targets: [{ format: HDR_FORMAT }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: DEPTH_FORMAT, depthCompare: 'greater', depthWriteEnabled: true },
    });
    this.group = bindGroup(device, layout, [...sky.shared, this.heightView]);
  }

  // Same bilinear lookup as terrainHeight() in WGSL.
  heightAt(x, z) {
    const res = TERRAIN_RES;
    const u = (x + TERRAIN_SIZE / 2) / TERRAIN_TEXEL - 0.5;
    const w = (z + TERRAIN_SIZE / 2) / TERRAIN_TEXEL - 0.5;
    const i = Math.floor(u);
    const j = Math.floor(w);
    const fx = u - i;
    const fz = w - j;
    const at = (a, b) => this.heights[Math.min(res - 1, Math.max(0, b)) * res + Math.min(res - 1, Math.max(0, a))];
    const top = at(i, j) + (at(i + 1, j) - at(i, j)) * fx;
    const bottom = at(i, j + 1) + (at(i + 1, j + 1) - at(i, j + 1)) * fx;
    return top + (bottom - top) * fz;
  }

  // A dry spot on the main island's beach, facing the open sea toward -Z.
  spawnPoint() {
    let z = ISLANDS[0].z;
    while (z > -TERRAIN_SIZE / 2 && this.heightAt(0, z) > 0.4) z -= 1;
    return [0, this.heightAt(0, z + 8), z + 8];
  }

  draw(pass) {
    if (!this.tileCount) return;
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group);
    pass.setVertexBuffer(0, this.tileBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint16');
    pass.drawIndexed(this.indexCount, this.tileCount);
  }
}
