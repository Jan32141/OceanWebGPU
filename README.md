# Ocean WebGPU

A real-time WebGPU ocean renderer written from scratch with no engine and no dependencies.

- **FFT waves**: Tessendorf ocean with 3 cascades (757 m / 109 m / 19 m, 256²). Uses a JONSWAP spectrum with Donelan-Banner spreading and finite-depth dispersion. Each frame runs a shared-memory radix-2 IFFT and produces choppy displacement and analytic slopes.
- **Whitecaps**: foam comes from the Jacobian of the horizontal displacement, kept over time and broken up with procedural noise.
- **Sun glitter**: GGX specular on per-pixel normals. Roughness increases with pixel footprint, and the reflections come from a mipmapped sky and cloud environment map.
- **Sky**: physically based atmosphere (Rayleigh, Mie and ozone) using a transmittance LUT and a per-frame sky-view LUT, plus a limb-darkened sun disk.
- **Clouds**: a 2.5D cumulus layer with a light march toward the sun and approximate multiple scattering, a cirrus layer, aerial perspective and cloud shadows on the water.
- **Post**: HDR, dual-filter bloom, ACES tonemapping and dithering.
- **Free camera** on a camera-centred polar grid, with reverse-Z infinite projection.

## Run

ES modules require a local server:

```sh
npx serve .        # or: python3 -m http.server
```

Then open the printed URL in a WebGPU browser (Chrome/Edge 113+, Safari 26+).

## Controls

| Input | Action |
| --- | --- |
| Drag | Look |
| WASD / arrows | Move |
| Space/E, C/Q | Up / down |
| Shift | Fast |
| Wheel | Base speed |
| H | Hide UI |

The panel adjusts the sun, wind and spectrum, choppiness, foam, clouds, exposure and bloom.

## Layout

```
src/main.js       frame loop, globals uniform
src/ocean.js      FFT simulation (spectrum, IFFT, assemble, mips)
src/surface.js    polar grid + ocean shading pipeline
src/sky.js        atmosphere LUTs, environment map, background sky
src/post.js       bloom + tonemap
src/shaders/      WGSL sources
```
