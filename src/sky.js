import { transmittanceWGSL, skyViewWGSL, envWGSL, backgroundWGSL } from './shaders/sky.js';
import { uniform, texture, storage, sampler, bindGroup, computePipeline, shaderModule, COMPUTE, groups } from './gpu.js';

export const HDR_FORMAT = 'rgba16float';
export const DEPTH_FORMAT = 'depth32float';

// Bindings 0..4 shared by every pass that shades sky light (sky, env, ocean).
export const skyLayoutEntries = () => [uniform(0), texture(1), texture(2), sampler(3), sampler(4)];

export class Sky {
  constructor(device, globalsBuffer, samplers, mipgen) {
    this.device = device;
    this.mipgen = mipgen;
    const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    this.transLUT = device.createTexture({ label: 'transmittance', size: [256, 64], format: 'rgba16float', usage });
    this.skyLUT = device.createTexture({ label: 'sky view', size: [256, 128], format: 'rgba16float', usage });
    this.env = device.createTexture({
      label: 'environment',
      size: [512, 256, 1],
      format: 'rgba16float',
      usage,
      mipLevelCount: 8,
    });
    this.envView = this.env.createView({ dimension: '2d-array' });
    this.envChain = mipgen.createChain(this.env);

    const transLayout = device.createBindGroupLayout({ entries: [storage(0, 'rgba16float')] });
    this.transPipeline = computePipeline(device, 'transmittance', transmittanceWGSL, 'main', [transLayout]);
    this.transGroup = bindGroup(device, transLayout, [this.transLUT.createView()]);

    const viewLayout = device.createBindGroupLayout({
      entries: [uniform(0, COMPUTE), texture(1, { visibility: COMPUTE }), sampler(2, COMPUTE), storage(3, 'rgba16float')],
    });
    this.viewPipeline = computePipeline(device, 'sky view', skyViewWGSL, 'main', [viewLayout]);
    this.viewGroup = bindGroup(device, viewLayout, [
      globalsBuffer,
      this.transLUT.createView(),
      samplers.clamp,
      this.skyLUT.createView(),
    ]);

    // Resources every sky-lit pass binds at 0..4.
    this.shared = [globalsBuffer, this.transLUT.createView(), this.skyLUT.createView(), samplers.clamp, samplers.wrapU];

    const envLayout = device.createBindGroupLayout({
      entries: [...skyLayoutEntries(), storage(5, 'rgba16float', '2d-array')],
    });
    this.envPipeline = computePipeline(device, 'environment', envWGSL, 'main', [envLayout]);
    this.envGroup = bindGroup(device, envLayout, [
      ...this.shared,
      this.env.createView({ dimension: '2d-array', baseMipLevel: 0, mipLevelCount: 1 }),
    ]);

    const bgLayout = device.createBindGroupLayout({ entries: skyLayoutEntries() });
    const bgModule = shaderModule(device, 'background', backgroundWGSL);
    this.bgPipeline = device.createRenderPipeline({
      label: 'background',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgLayout] }),
      vertex: { module: bgModule, entryPoint: 'vs' },
      fragment: { module: bgModule, entryPoint: 'fs', targets: [{ format: HDR_FORMAT }] },
      // Reverse-Z: only pixels still at the cleared far depth (0) get sky.
      depthStencil: { format: DEPTH_FORMAT, depthCompare: 'equal', depthWriteEnabled: false },
    });
    this.bgGroup = bindGroup(device, bgLayout, this.shared);

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.transPipeline);
    pass.setBindGroup(0, this.transGroup);
    pass.dispatchWorkgroups(groups(256, 8), groups(64, 8));
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  encode(pass) {
    pass.setPipeline(this.viewPipeline);
    pass.setBindGroup(0, this.viewGroup);
    pass.dispatchWorkgroups(groups(256, 8), groups(128, 8));
    pass.setPipeline(this.envPipeline);
    pass.setBindGroup(0, this.envGroup);
    pass.dispatchWorkgroups(groups(512, 8), groups(256, 8));
    this.mipgen.encode(pass, this.envChain);
  }

  draw(pass) {
    pass.setPipeline(this.bgPipeline);
    pass.setBindGroup(0, this.bgGroup);
    pass.draw(3);
  }
}
