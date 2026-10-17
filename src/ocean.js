import { CASCADES, FFT_SIZE } from './shaders/common.js';
import { initSpectrumWGSL, timeSpectrumWGSL, fftWGSL, assembleWGSL } from './shaders/simulation.js';
import { uniform, texture, storage, bindGroup, computePipeline, COMPUTE, groups } from './gpu.js';

const N = FFT_SIZE;

// Patch sizes (m). Each cascade only holds wavelengths between its neighbours'
// cut-offs so energy is not counted twice; incommensurate sizes hide tiling.
export const CASCADE_LENGTHS = [757, 109, 19];

// GPU FFT ocean (Tessendorf) with three cascades.
// Per frame: evolve spectrum -> 2D inverse FFT -> displacement/slope/foam maps -> mips.
export class OceanSimulation {
  constructor(device, mipgen) {
    this.device = device;
    this.mipgen = mipgen;
    this.params = new Float32Array(24);
    this.paramBuffer = device.createBuffer({
      label: 'sim params',
      size: this.params.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.dirty = true;

    const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    const arrayTex = (label, format, layers, mipLevelCount = 1) =>
      device.createTexture({ label, size: [N, N, layers], format, usage, mipLevelCount });
    const mips = Math.log2(N) + 1;

    this.h0 = arrayTex('h0', 'rgba32float', CASCADES);
    this.fftA = arrayTex('fftA', 'rgba32float', CASCADES * 2);
    this.fftB = arrayTex('fftB', 'rgba32float', CASCADES * 2);
    // Displacement (xyz) + foam (w) is double-buffered so foam can decay over time.
    this.disp = [arrayTex('disp0', 'rgba16float', CASCADES, mips), arrayTex('disp1', 'rgba16float', CASCADES, mips)];
    this.deriv = arrayTex('deriv', 'rgba16float', CASCADES, mips);

    const arr = { dimension: '2d-array' };
    const level0 = { dimension: '2d-array', baseMipLevel: 0, mipLevelCount: 1 };
    const unfilt = { dim: '2d-array', type: 'unfilterable-float', visibility: COMPUTE };

    const initLayout = device.createBindGroupLayout({
      entries: [uniform(0, COMPUTE), storage(1, 'rgba32float', '2d-array')],
    });
    const timeLayout = device.createBindGroupLayout({
      entries: [uniform(0, COMPUTE), texture(1, unfilt), storage(2, 'rgba32float', '2d-array')],
    });
    const fftLayout = device.createBindGroupLayout({
      entries: [texture(0, unfilt), storage(1, 'rgba32float', '2d-array')],
    });
    const assembleLayout = device.createBindGroupLayout({
      entries: [
        uniform(0, COMPUTE),
        texture(1, unfilt),
        texture(2, unfilt),
        storage(3, 'rgba16float', '2d-array'),
        storage(4, 'rgba16float', '2d-array'),
      ],
    });

    this.initPipeline = computePipeline(device, 'init spectrum', initSpectrumWGSL, 'main', [initLayout]);
    this.timePipeline = computePipeline(device, 'time spectrum', timeSpectrumWGSL, 'main', [timeLayout]);
    this.fftHPipeline = computePipeline(device, 'fft horizontal', fftWGSL, 'horizontal', [fftLayout]);
    this.fftVPipeline = computePipeline(device, 'fft vertical', fftWGSL, 'vertical', [fftLayout]);
    this.assemblePipeline = computePipeline(device, 'assemble', assembleWGSL, 'main', [assembleLayout]);

    this.initGroup = bindGroup(device, initLayout, [this.paramBuffer, this.h0.createView(arr)]);
    this.timeGroup = bindGroup(device, timeLayout, [
      this.paramBuffer,
      this.h0.createView(arr),
      this.fftA.createView(arr),
    ]);
    this.fftHGroup = bindGroup(device, fftLayout, [this.fftA.createView(arr), this.fftB.createView(arr)]);
    this.fftVGroup = bindGroup(device, fftLayout, [this.fftB.createView(arr), this.fftA.createView(arr)]);
    this.assembleGroups = [0, 1].map((p) =>
      bindGroup(device, assembleLayout, [
        this.paramBuffer,
        this.fftA.createView(arr),
        this.disp[1 - p].createView(level0),
        this.disp[p].createView(level0),
        this.deriv.createView(level0),
      ]),
    );
    this.dispChains = this.disp.map((t) => mipgen.createChain(t));
    this.derivChain = mipgen.createChain(this.deriv);

    this.dispViews = this.disp.map((t) => t.createView(arr));
    this.derivView = this.deriv.createView(arr);
  }

  // Spectrum-shaping parameters; changing them rebuilds h0.
  setSpectrum({ windSpeed, windDir, fetch, spread, depth = 500, seed = 1 }) {
    const p = this.params;
    p[12] = windSpeed;
    p[13] = windDir;
    p[14] = fetch;
    p[15] = 9.81;
    p[16] = depth;
    p[22] = spread;
    p[23] = seed;
    this.dirty = true;
  }

  update(time, dt, { choppiness, foamThreshold, foamDecay }) {
    const p = this.params;
    const L = CASCADE_LENGTHS;
    p.set([L[0], L[1], L[2], 0], 0);
    // Cascade i keeps wavelengths longer than L[i+1] / 6.
    const cut = (i) => (i < CASCADES - 1 ? ((2 * Math.PI) / L[i + 1]) * 6 : 1e9);
    p.set([1e-4, cut(0), cut(1), 0], 4);
    p.set([cut(0), cut(1), cut(2), 0], 8);
    p[17] = time;
    p[18] = dt;
    p[19] = choppiness;
    p[20] = foamThreshold;
    p[21] = foamDecay;
    this.device.queue.writeBuffer(this.paramBuffer, 0, p);
  }

  encode(pass, parity) {
    const tiles = groups(N, 8);
    if (this.dirty) {
      pass.setPipeline(this.initPipeline);
      pass.setBindGroup(0, this.initGroup);
      pass.dispatchWorkgroups(tiles, tiles, CASCADES);
      this.dirty = false;
    }
    pass.setPipeline(this.timePipeline);
    pass.setBindGroup(0, this.timeGroup);
    pass.dispatchWorkgroups(tiles, tiles, CASCADES);

    pass.setPipeline(this.fftHPipeline);
    pass.setBindGroup(0, this.fftHGroup);
    pass.dispatchWorkgroups(N, CASCADES * 2);
    pass.setPipeline(this.fftVPipeline);
    pass.setBindGroup(0, this.fftVGroup);
    pass.dispatchWorkgroups(N, CASCADES * 2);

    pass.setPipeline(this.assemblePipeline);
    pass.setBindGroup(0, this.assembleGroups[parity]);
    pass.dispatchWorkgroups(tiles, tiles, CASCADES);

    this.mipgen.encode(pass, this.dispChains[parity]);
    this.mipgen.encode(pass, this.derivChain);
  }
}
