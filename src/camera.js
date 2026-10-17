import { directionFromAngles, normalize, cross } from './math.js';

// Free-fly camera: drag to look, WASD to move, Space/C (or E/Q) for up/down,
// Shift to go faster, mouse wheel to change base speed.
export class FreeCamera {
  constructor(canvas) {
    this.position = [0, 6, 0];
    this.yaw = 0;
    this.pitch = 0.03;
    this.speed = 12;
    this.fovY = (55 * Math.PI) / 180;
    this.keys = new Set();
    this.velocity = [0, 0, 0];

    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    canvas.addEventListener('pointerdown', (e) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointerup', (e) => {
      dragging = false;
      canvas.releasePointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const scale = this.fovY / canvas.clientHeight;
      this.yaw += (e.clientX - lastX) * scale;
      this.pitch -= (e.clientY - lastY) * scale;
      this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch));
      lastX = e.clientX;
      lastY = e.clientY;
    });
    canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        this.speed *= Math.pow(1.15, -Math.sign(e.deltaY));
        this.speed = Math.max(0.5, Math.min(3000, this.speed));
      },
      { passive: false },
    );
    window.addEventListener('keydown', (e) => {
      if (e.target instanceof HTMLInputElement) return;
      this.keys.add(e.code);
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => this.keys.clear());
  }

  get forward() {
    return directionFromAngles(this.yaw, this.pitch);
  }

  update(dt) {
    const f = this.forward;
    const r = normalize(cross(f, [0, 1, 0]));
    const k = this.keys;
    const move = [0, 0, 0];
    const add = (v, s) => {
      move[0] += v[0] * s;
      move[1] += v[1] * s;
      move[2] += v[2] * s;
    };
    if (k.has('KeyW') || k.has('ArrowUp')) add(f, 1);
    if (k.has('KeyS') || k.has('ArrowDown')) add(f, -1);
    if (k.has('KeyD') || k.has('ArrowRight')) add(r, 1);
    if (k.has('KeyA') || k.has('ArrowLeft')) add(r, -1);
    if (k.has('Space') || k.has('KeyE')) add([0, 1, 0], 1);
    if (k.has('KeyC') || k.has('KeyQ')) add([0, 1, 0], -1);

    const boost = k.has('ShiftLeft') || k.has('ShiftRight') ? 6 : 1;
    const target = move.map((m) => m * this.speed * boost);
    // Exponential smoothing gives the camera a little inertia.
    const a = 1 - Math.exp(-dt * 8);
    for (let i = 0; i < 3; i++) {
      this.velocity[i] += (target[i] - this.velocity[i]) * a;
      this.position[i] += this.velocity[i] * dt;
    }
    this.position[1] = Math.max(1.5, Math.min(20000, this.position[1]));
  }
}
