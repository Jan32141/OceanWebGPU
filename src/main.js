import { FreeCamera } from './camera.js';
import { perspectiveReverseZ, viewRotation, multiply, invert } from './math.js';
import { MipGenerator } from './mipgen.js';
import { OceanSimulation, CASCADE_LENGTHS } from './ocean.js';
import { Sky } from './sky.js';
import { OceanSurface } from './surface.js';
import { PostProcess } from './post.js';
import { setupUI } from './ui.js';

const DEG = Math.PI / 180;

function showError(message) {
  const el = document.getElementById('error');
  el.textContent = message;
  el.hidden = false;
}

async function init() {
  if (!navigator.gpu) {
    showError('WebGPU is not available in this browser.\nTry a recent Chrome, Edge, or Safari 26+.');
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    showError('No suitable GPU adapter was found.');
    return;
  }
  const device = await adapter.requestDevice();
  device.lost.then((info) => info.reason !== 'destroyed' && showError(`GPU device lost: ${info.message}`));
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU error:', e.error.message));

  const canvas = document.getElementById('canvas');
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const linear = { magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' };
  const samplers = {
    repeat: device.createSampler({ ...linear, addressModeU: 'repeat', addressModeV: 'repeat', maxAnisotropy: 16 }),
    wrapU: device.createSampler({ ...linear, addressModeU: 'repeat', addressModeV: 'clamp-to-edge' }),
    clamp: device.createSampler({ ...linear, addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' }),
  };

  // Layout mirrors `struct Globals` in shaders/common.js.
  const globals = new Float32Array(56);
  const globalsBuffer = device.createBuffer({
    label: 'globals',
    size: globals.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const mipgen = new MipGenerator(device);
  const ocean = new OceanSimulation(device, mipgen);
  const sky = new Sky(device, globalsBuffer, samplers, mipgen);
  const surface = new OceanSurface(device, sky, ocean, samplers);
  const post = new PostProcess(device, globalsBuffer, samplers, format);
  const camera = new FreeCamera(canvas);

  const applySpectrum = (v) =>
    ocean.setSpectrum({ windSpeed: v.windSpeed, windDir: v.windDir * DEG, fetch: v.fetch * 1000, spread: v.spread });
  const params = setupUI(applySpectrum);
  applySpectrum(params);

  const fpsEl = document.getElementById('fps');
  let frames = 0;
  let fpsTime = performance.now();
  let last = performance.now();
  let simTime = 0;
  let cloudTime = 0;
  let parity = 0;

  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    const simDt = dt * params.timeScale;
    simTime += simDt;
    cloudTime += dt * params.cloudSpeed;
    camera.update(dt);

    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    post.resize(w, h);

    const proj = perspectiveReverseZ(camera.fovY, w / h, 0.1);
    const viewProj = multiply(proj, viewRotation(camera.forward));
    const el = params.sunElevation * DEG;
    const az = params.sunAzimuth * DEG;
    const wind = params.windDir * DEG;

    globals.set(viewProj, 0);
    globals.set(invert(viewProj), 16);
    globals.set(camera.position, 32);
    globals[35] = simTime;
    globals.set([Math.cos(el) * Math.sin(az), Math.sin(el), -Math.cos(el) * Math.cos(az)], 36);
    globals[39] = params.sunIntensity;
    globals.set([...CASCADE_LENGTHS, params.choppiness], 40);
    globals.set([w, h, params.exposure, params.bloom], 44);
    globals[48] = params.cloudCoverage;
    globals[49] = params.cloudDensity;
    // Clouds drift downwind (noise space units).
    globals[50] = -Math.cos(wind) * cloudTime * 0.004;
    globals[51] = -Math.sin(wind) * cloudTime * 0.004;
    globals[52] = params.foamAmount;
    globals[53] = params.cirrus;
    device.queue.writeBuffer(globalsBuffer, 0, globals);

    ocean.update(simTime, simDt, {
      choppiness: params.choppiness,
      foamThreshold: params.foamCoverage,
      foamDecay: params.foamDecay,
    });

    const encoder = device.createCommandEncoder();
    const compute = encoder.beginComputePass({ label: 'simulation + sky' });
    ocean.encode(compute, parity);
    sky.encode(compute);
    compute.end();

    const scene = encoder.beginRenderPass({
      label: 'scene',
      colorAttachments: [{ view: post.hdrView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      depthStencilAttachment: {
        view: post.depthView,
        depthClearValue: 0,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    });
    surface.draw(scene, parity);
    sky.draw(scene);
    scene.end();

    post.encode(encoder, context.getCurrentTexture().createView());
    device.queue.submit([encoder.finish()]);
    parity ^= 1;

    frames++;
    if (now - fpsTime > 500) {
      fpsEl.textContent = `${Math.round((frames * 1000) / (now - fpsTime))} fps`;
      frames = 0;
      fpsTime = now;
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

init().catch((e) => {
  console.error(e);
  showError(`Failed to start: ${e.message}`);
});
