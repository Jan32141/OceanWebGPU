import { bloomDownWGSL, bloomUpWGSL, tonemapWGSL } from './shaders/post.js';
import { uniform, texture, storage, sampler, bindGroup, computePipeline, shaderModule, COMPUTE, groups } from './gpu.js';
import { HDR_FORMAT, DEPTH_FORMAT } from './sky.js';

const MAX_BLOOM_LEVELS = 6;

// Owns the HDR target, depth buffer, bloom chain and the final tonemap pass.
export class PostProcess {
  constructor(device, globalsBuffer, samplers, canvasFormat) {
    this.device = device;
    this.globalsBuffer = globalsBuffer;
    this.samplers = samplers;
    this.width = 0;
    this.height = 0;
    this.levelBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const vis = { visibility: COMPUTE };
    this.downLayout = device.createBindGroupLayout({
      entries: [texture(0, vis), storage(1, 'rgba16float'), sampler(2, COMPUTE)],
    });
    this.upLayout = device.createBindGroupLayout({
      entries: [texture(0, vis), texture(1, vis), storage(2, 'rgba16float'), sampler(3, COMPUTE)],
    });
    this.firstPipeline = computePipeline(device, 'bloom first', bloomDownWGSL, 'first', [this.downLayout]);
    this.downPipeline = computePipeline(device, 'bloom down', bloomDownWGSL, 'down', [this.downLayout]);
    this.upPipeline = computePipeline(device, 'bloom up', bloomUpWGSL, 'main', [this.upLayout]);

    this.toneLayout = device.createBindGroupLayout({
      entries: [uniform(0), texture(1), texture(2), sampler(3), uniform(4)],
    });
    const module = shaderModule(device, 'tonemap', tonemapWGSL);
    this.tonePipeline = device.createRenderPipeline({
      label: 'tonemap',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.toneLayout] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: canvasFormat }] },
    });
  }

  resize(width, height) {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    const d = this.device;
    for (const t of [this.hdr, this.depth, this.bloomDown, this.bloomUp]) t?.destroy();

    this.hdr = d.createTexture({
      size: [width, height],
      format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depth = d.createTexture({ size: [width, height], format: DEPTH_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.hdrView = this.hdr.createView();
    this.depthView = this.depth.createView();

    const bw = Math.max(1, width >> 1);
    const bh = Math.max(1, height >> 1);
    this.levels = Math.max(1, Math.min(MAX_BLOOM_LEVELS, Math.floor(Math.log2(Math.min(bw, bh))) - 1));
    const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    const chain = (label) => d.createTexture({ label, size: [bw, bh], format: 'rgba16float', usage, mipLevelCount: this.levels });
    this.bloomDown = chain('bloom down');
    this.bloomUp = chain('bloom up');
    const mip = (t, l) => t.createView({ baseMipLevel: l, mipLevelCount: 1 });
    const size = (l) => [Math.max(1, bw >> l), Math.max(1, bh >> l)];
    const s = this.samplers.clamp;

    this.downSteps = [];
    for (let l = 0; l < this.levels; l++) {
      const src = l === 0 ? this.hdrView : mip(this.bloomDown, l - 1);
      this.downSteps.push({
        pipeline: l === 0 ? this.firstPipeline : this.downPipeline,
        group: bindGroup(d, this.downLayout, [src, mip(this.bloomDown, l), s]),
        size: size(l),
      });
    }
    // up[l] = down[l] + tent(up[l+1]); the smallest level starts from down[last].
    this.upSteps = [];
    const last = this.levels - 1;
    for (let l = last - 1; l >= 0; l--) {
      const small = l === last - 1 ? mip(this.bloomDown, last) : mip(this.bloomUp, l + 1);
      this.upSteps.push({
        group: bindGroup(d, this.upLayout, [small, mip(this.bloomDown, l), mip(this.bloomUp, l), s]),
        size: size(l),
      });
    }
    const bloomResult = this.levels > 1 ? mip(this.bloomUp, 0) : mip(this.bloomDown, 0);
    d.queue.writeBuffer(this.levelBuffer, 0, new Float32Array([this.levels, 0, 0, 0]));
    this.toneGroup = bindGroup(d, this.toneLayout, [this.globalsBuffer, this.hdrView, bloomResult, s, this.levelBuffer]);
  }

  encode(encoder, targetView) {
    const pass = encoder.beginComputePass({ label: 'bloom' });
    for (const st of this.downSteps) {
      pass.setPipeline(st.pipeline);
      pass.setBindGroup(0, st.group);
      pass.dispatchWorkgroups(groups(st.size[0], 8), groups(st.size[1], 8));
    }
    pass.setPipeline(this.upPipeline);
    for (const st of this.upSteps) {
      pass.setBindGroup(0, st.group);
      pass.dispatchWorkgroups(groups(st.size[0], 8), groups(st.size[1], 8));
    }
    pass.end();

    const rp = encoder.beginRenderPass({
      label: 'tonemap',
      colorAttachments: [{ view: targetView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    rp.setPipeline(this.tonePipeline);
    rp.setBindGroup(0, this.toneGroup);
    rp.draw(3);
    rp.end();
  }
}
