import { surfaceWGSL } from './shaders/surface.js';
import { texture, sampler, bindGroup, shaderModule } from './gpu.js';
import { skyLayoutEntries, HDR_FORMAT, DEPTH_FORMAT } from './sky.js';

// Camera-centred polar grid: ring radii grow geometrically so vertex density
// tracks screen-space density, with a final far ring out to the horizon.
function buildGrid(rings = 512, segments = 512, first = 0.06, radius = 40000, farRadius = 1e6) {
  const n = rings - 2;
  const sum = (q) => (first * (Math.pow(q, n) - 1)) / (q - 1);
  let lo = 1.000001;
  let hi = 1.5;
  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (sum(mid) > radius) hi = mid;
    else lo = mid;
  }
  const q = (lo + hi) / 2;

  const verts = new Float32Array(rings * segments * 3);
  let o = 0;
  for (let i = 0; i < rings; i++) {
    const last = i === rings - 1;
    const r = last ? farRadius : (first * (Math.pow(q, i) - 1)) / (q - 1);
    const spacing = last ? farRadius : Math.max(first * Math.pow(q, i), (r * 2 * Math.PI) / segments);
    for (let j = 0; j < segments; j++) {
      const a = (j / segments) * 2 * Math.PI;
      verts[o++] = Math.cos(a) * r;
      verts[o++] = Math.sin(a) * r;
      verts[o++] = spacing;
    }
  }
  const indices = new Uint32Array((rings - 1) * segments * 6);
  o = 0;
  for (let i = 0; i < rings - 1; i++) {
    for (let j = 0; j < segments; j++) {
      const j1 = (j + 1) % segments;
      const a = i * segments + j;
      const b = i * segments + j1;
      const c = (i + 1) * segments + j;
      const d = (i + 1) * segments + j1;
      indices.set([a, c, b, b, c, d], o);
      o += 6;
    }
  }
  return { verts, indices };
}

export class OceanSurface {
  constructor(device, sky, ocean, samplers) {
    const { verts, indices } = buildGrid();
    this.vertexBuffer = device.createBuffer({
      label: 'ocean grid',
      size: verts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.vertexBuffer, 0, verts);
    this.indexBuffer = device.createBuffer({
      label: 'ocean indices',
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.indexBuffer, 0, indices);
    this.indexCount = indices.length;

    const arr = { dim: '2d-array' };
    const layout = device.createBindGroupLayout({
      label: 'ocean surface',
      entries: [...skyLayoutEntries(), texture(5, arr), texture(6, arr), texture(7, arr), sampler(8)],
    });
    const module = shaderModule(device, 'ocean surface', surfaceWGSL);
    this.pipeline = device.createRenderPipeline({
      label: 'ocean surface',
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: {
        module,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 12,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x2' },
              { shaderLocation: 1, offset: 8, format: 'float32' },
            ],
          },
        ],
      },
      fragment: { module, entryPoint: 'fs', targets: [{ format: HDR_FORMAT }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: DEPTH_FORMAT, depthCompare: 'greater', depthWriteEnabled: true },
    });
    this.groups = [0, 1].map((p) =>
      bindGroup(device, layout, [
        ...sky.shared,
        ocean.dispViews[p],
        ocean.derivView,
        sky.envView,
        samplers.repeat,
      ]),
    );
  }

  draw(pass, parity) {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.groups[parity]);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
    pass.drawIndexed(this.indexCount);
  }
}
