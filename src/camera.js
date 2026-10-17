import { directionFromAngles, normalize, cross } from './math.js';

// Camera with two modes, toggled with F:
//  walk - WASD on the ground, Space to jump, Shift to run; wades and swims in water.
//  fly  - free flight, Space/C (or E/Q) for up/down, Shift faster, wheel for speed.
const EYE_HEIGHT = 1.7;
const GRAVITY = 22;
const JUMP_SPEED = 7;

export class FreeCamera {
  constructor(canvas) {
    this.position = [0, 6, 0];
    this.yaw = 0;
    this.pitch = 0.03;
    this.speed = 12;
    this.fovY = (55 * Math.PI) / 180;
    this.keys = new Set();
    this.velocity = [0, 0, 0];
    this.walking = true;
    this.onGround = false;
    this.onModeChange = null;
    this.onLockChange = null;

    const look = (dx, dy, scale) => {
      this.yaw += dx * scale;
      this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch - dy * scale));
    };
    // Mouse: click to capture the pointer, then just move the mouse to look (Esc releases).
    this.locked = false;
    canvas.addEventListener('click', () => {
      if (!this.locked && canvas.requestPointerLock) canvas.requestPointerLock();
    });
    document.addEventListener('pointerlockchange', () => {
      this.locked = document.pointerLockElement === canvas;
      this.onLockChange?.(this.locked);
    });
    document.addEventListener('mousemove', (e) => {
      if (this.locked) look(e.movementX, e.movementY, 0.0022);
    });

    // Touch: drag to look.
    let touchId = null;
    let lastX = 0;
    let lastY = 0;
    canvas.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'mouse') return;
      touchId = e.pointerId;
      lastX = e.clientX;
      lastY = e.clientY;
    });
    canvas.addEventListener('pointerup', (e) => {
      if (e.pointerId === touchId) touchId = null;
    });
    canvas.addEventListener('pointermove', (e) => {
      if (e.pointerId !== touchId) return;
      look(e.clientX - lastX, e.clientY - lastY, this.fovY / canvas.clientHeight);
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
      if (e.code === 'KeyF') {
        this.walking = !this.walking;
        this.velocity[1] = 0;
        this.onModeChange?.(this.walking);
      }
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => this.keys.clear());
  }

  get forward() {
    return directionFromAngles(this.yaw, this.pitch);
  }

  // groundAt(x, z) returns terrain height (m); negative below sea level.
  update(dt, groundAt) {
    if (this.walking) this.walk(dt, groundAt);
    else this.fly(dt, groundAt);
  }

  walk(dt, groundAt) {
    const k = this.keys;
    const f = [Math.sin(this.yaw), 0, -Math.cos(this.yaw)];
    const r = [-f[2], 0, f[0]];
    let mx = 0;
    let mz = 0;
    if (k.has('KeyW') || k.has('ArrowUp')) { mx += f[0]; mz += f[2]; }
    if (k.has('KeyS') || k.has('ArrowDown')) { mx -= f[0]; mz -= f[2]; }
    if (k.has('KeyD') || k.has('ArrowRight')) { mx += r[0]; mz += r[2]; }
    if (k.has('KeyA') || k.has('ArrowLeft')) { mx -= r[0]; mz -= r[2]; }
    const len = Math.hypot(mx, mz) || 1;

    const p = this.position;
    const ground = groundAt(p[0], p[2]);
    // Stand on the ground, wade through shallows, or float with the head above water.
    const swimming = ground + EYE_HEIGHT < 0.45;
    const run = k.has('ShiftLeft') || k.has('ShiftRight');
    const speed = (run ? 9 : 4.5) * (swimming ? 0.45 : ground < 0 ? 0.7 : 1);

    const a = 1 - Math.exp(-dt * (this.onGround || swimming ? 12 : 2));
    this.velocity[0] += ((mx / len) * speed - this.velocity[0]) * a;
    this.velocity[2] += ((mz / len) * speed - this.velocity[2]) * a;

    if (this.onGround && (k.has('Space') || k.has('KeyE'))) this.velocity[1] = swimming ? 2.5 : JUMP_SPEED;
    this.velocity[1] -= GRAVITY * dt;

    p[0] += this.velocity[0] * dt;
    p[2] += this.velocity[2] * dt;
    p[1] += this.velocity[1] * dt;

    const newFloor = Math.max(groundAt(p[0], p[2]) + EYE_HEIGHT, 0.45);
    if (p[1] <= newFloor) {
      p[1] = newFloor;
      this.velocity[1] = 0;
      this.onGround = true;
    } else {
      // Stay glued when walking downhill instead of hopping off every bump.
      this.onGround = this.onGround && this.velocity[1] <= 0 && p[1] - newFloor < 0.6;
      if (this.onGround) p[1] = newFloor;
    }
  }

  fly(dt, groundAt) {
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
    const p = this.position;
    p[1] = Math.max(Math.max(groundAt(p[0], p[2]), 0) + 1.5, Math.min(20000, p[1]));
  }
}
