import { mipWGSL } from './shaders/simulation.js';
import { texture, storage, bindGroup, computePipeline, groups } from './gpu.js';

// Builds box-filtered mip chains for rgba16float 2d-array textures in a compute pass.
export class MipGenerator {
  constructor(device) {
    this.device = device;
    this.layout = device.createBindGroupLayout({
      label: 'mipgen',
      entries: [texture(0, { dim: '2d-array', type: 'unfilterable-float' }), storage(1, 'rgba16float', '2d-array')],
    });
    this.pipeline = computePipeline(device, 'mipgen', mipWGSL, 'main', [this.layout]);
  }

  createChain(tex) {
    const steps = [];
    for (let level = 1; level < tex.mipLevelCount; level++) {
      const view = (l) => tex.createView({ dimension: '2d-array', baseMipLevel: l, mipLevelCount: 1 });
      steps.push({
        group: bindGroup(this.device, this.layout, [view(level - 1), view(level)]),
        width: Math.max(1, tex.width >> level),
        height: Math.max(1, tex.height >> level),
        layers: tex.depthOrArrayLayers,
      });
    }
    return steps;
  }

  encode(pass, chain) {
    pass.setPipeline(this.pipeline);
    for (const s of chain) {
      pass.setBindGroup(0, s.group);
      pass.dispatchWorkgroups(groups(s.width, 8), groups(s.height, 8), s.layers);
    }
  }
}
