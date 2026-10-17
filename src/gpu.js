// Small helpers for building explicit bind group layouts and pipelines.
// Stage flags are numeric so this module loads even where WebGPU is missing.

export const VERTEX = 1;
export const FRAGMENT = 2;
export const COMPUTE = 4;
export const ALL = VERTEX | FRAGMENT | COMPUTE;

export const uniform = (binding, visibility = ALL) => ({ binding, visibility, buffer: { type: 'uniform' } });

export const texture = (binding, { dim = '2d', type = 'float', visibility = ALL } = {}) => ({
  binding,
  visibility,
  texture: { sampleType: type, viewDimension: dim },
});

export const storage = (binding, format, dim = '2d') => ({
  binding,
  visibility: COMPUTE,
  storageTexture: { access: 'write-only', format, viewDimension: dim },
});

export const sampler = (binding, visibility = ALL) => ({ binding, visibility, sampler: { type: 'filtering' } });

export function bindGroup(device, layout, resources, label) {
  return device.createBindGroup({
    label,
    layout,
    entries: resources.map((resource, binding) =>
      resource instanceof GPUBuffer ? { binding, resource: { buffer: resource } } : { binding, resource },
    ),
  });
}

export function shaderModule(device, label, code) {
  const module = device.createShaderModule({ label, code });
  module.getCompilationInfo().then((info) => {
    for (const m of info.messages) {
      const log = m.type === 'error' ? console.error : console.warn;
      log(`[${label}] ${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
    }
  });
  return module;
}

export function computePipeline(device, label, code, entryPoint, layouts) {
  return device.createComputePipeline({
    label,
    layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
    compute: { module: shaderModule(device, label, code), entryPoint },
  });
}

export const groups = (n, size) => Math.ceil(n / size);
