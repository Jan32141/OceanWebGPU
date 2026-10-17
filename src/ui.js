// Parameter panel. `spectrum: true` entries rebuild the wave spectrum when changed.
export const PARAMS = [
  { group: 'Sun' },
  { key: 'sunElevation', label: 'Elevation', min: -4, max: 60, step: 0.1, value: 3.2, unit: '°' },
  { key: 'sunAzimuth', label: 'Azimuth', min: -180, max: 180, step: 0.5, value: 0, unit: '°' },
  { key: 'sunIntensity', label: 'Intensity', min: 5, max: 40, step: 0.5, value: 20 },
  { group: 'Waves' },
  { key: 'windSpeed', label: 'Wind speed', min: 2, max: 30, step: 0.5, value: 11, unit: ' m/s', spectrum: true },
  { key: 'windDir', label: 'Wind direction', min: -180, max: 180, step: 1, value: 70, unit: '°', spectrum: true },
  { key: 'fetch', label: 'Fetch', min: 10, max: 1000, step: 10, value: 400, unit: ' km', spectrum: true },
  { key: 'spread', label: 'Directional spread', min: 0, max: 1, step: 0.01, value: 0.15, spectrum: true },
  { key: 'choppiness', label: 'Choppiness', min: 0, max: 1.6, step: 0.01, value: 0.95 },
  { key: 'timeScale', label: 'Time scale', min: 0, max: 3, step: 0.05, value: 1 },
  { group: 'Foam' },
  { key: 'foamCoverage', label: 'Whitecap threshold', min: 0, max: 1.2, step: 0.01, value: 0.55 },
  { key: 'foamAmount', label: 'Foam intensity', min: 0, max: 2, step: 0.01, value: 1 },
  { key: 'foamDecay', label: 'Foam decay', min: 0.05, max: 3, step: 0.05, value: 0.5 },
  { group: 'Clouds' },
  { key: 'cloudCoverage', label: 'Coverage', min: 0, max: 1, step: 0.01, value: 0.42 },
  { key: 'cloudDensity', label: 'Density', min: 0.1, max: 3, step: 0.05, value: 1 },
  { key: 'cirrus', label: 'Cirrus', min: 0, max: 1, step: 0.01, value: 0.6 },
  { key: 'cloudSpeed', label: 'Drift speed', min: 0, max: 5, step: 0.1, value: 1 },
  { group: 'Camera' },
  { key: 'exposure', label: 'Exposure', min: 0.1, max: 6, step: 0.05, value: 1.6 },
  { key: 'bloom', label: 'Bloom', min: 0, max: 0.3, step: 0.005, value: 0.06 },
];

export function setupUI(onSpectrumChange) {
  const values = {};
  const root = document.getElementById('controls');
  for (const p of PARAMS) {
    if (p.group) {
      const h = document.createElement('h2');
      h.textContent = p.group;
      root.append(h);
      continue;
    }
    values[p.key] = p.value;
    const row = document.createElement('div');
    row.className = 'row';
    const label = document.createElement('label');
    label.textContent = p.label;
    label.htmlFor = `p-${p.key}`;
    const out = document.createElement('output');
    const input = document.createElement('input');
    Object.assign(input, { type: 'range', id: `p-${p.key}`, min: p.min, max: p.max, step: p.step, value: p.value });
    const show = () => (out.textContent = `${Number(input.value).toFixed(p.step < 0.1 ? 2 : 1)}${p.unit ?? ''}`);
    input.addEventListener('input', () => {
      values[p.key] = Number(input.value);
      show();
      if (p.spectrum) onSpectrumChange(values);
    });
    show();
    row.append(label, out, input);
    root.append(row);
  }

  const panel = document.getElementById('panel');
  const toggle = document.getElementById('toggle');
  toggle.addEventListener('click', () => {
    panel.classList.toggle('collapsed');
    toggle.textContent = panel.classList.contains('collapsed') ? '+' : '–';
  });
  window.addEventListener('keydown', (e) => {
    if (e.code === 'KeyH' && !(e.target instanceof HTMLInputElement)) document.body.classList.toggle('no-ui');
  });
  return values;
}
