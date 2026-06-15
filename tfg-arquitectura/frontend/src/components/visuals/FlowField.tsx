import { useEffect, useRef } from "react";
import { Renderer, Program, Mesh, Triangle, Vec2, Vec3 } from "ogl";

/* ------------------------------------------------------------------------
 * FlowField - living WebGL background for the landing hero.
 *
 * A single full-screen triangle runs a hand-written GLSL shader: layered,
 * horizontally-drifting luminous bands (fbm domain-warp) that breathe and
 * never repeat - read as forecast trajectories / market flow, on-theme for
 * an inflation time-series thesis. Tinted to the project tokens:
 *   violet  #8B5CF6  (--mcp,  dominant)
 *   gold    #E0B96A  (--gold, reserved for the brightest crests)
 *   cyan    #06B6D4  (--info, cool accent)
 * over a near-black zinc base (#0A0A0C -> --background).
 *
 * Guards (all required):
 *   | prefers-reduced-motion -> render one static frame, no RAF loop
 *   | WebGL unavailable       -> render nothing (parent CSS gradient shows)
 *   | canvas offscreen        -> pause the RAF loop (IntersectionObserver)
 *
 * Inspired by the ogl-based shaders on 21st.dev (AuroraWaves / Plasma),
 * rewritten for this palette and concept rather than copied.
 * ------------------------------------------------------------------------ */

type FlowTheme = "dark" | "light" | "violet";

interface FlowFieldProps {
  className?: string;
  /** Flow speed multiplier. Calm by default. */
  speed?: number;
  /** Global opacity of the effect (the canvas itself). */
  opacity?: number;
  /** Subtle mouse parallax. Off by default to keep it ambient. */
  interactive?: boolean;
  /** Palette to render. Default dark. */
  theme?: FlowTheme;
}

const VERTEX = /* glsl */ `
  attribute vec2 position;
  void main() {
    gl_Position = vec4(position, 0.0, 1.0);
  }
`;

const FRAGMENT = /* glsl */ `
  precision highp float;

  uniform vec2  uResolution;
  uniform float uTime;
  uniform vec2  uMouse;        // 0..1, screen space (y up)
  uniform float uInteractive;  // 0 or 1
  uniform vec3  uBase;         // near-black zinc
  uniform vec3  uMcp;          // violet
  uniform vec3  uGold;         // champagne gold
  uniform vec3  uInfo;         // cyan
  uniform vec3  uIndigo;       // deep indigo (cloud mid-tone)
  uniform float uLight;        // 1.0 -> light theme (subtractive pastel fog)

  // -- value noise + fbm ----------------------------------------------
  float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
  }

  float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
    for (int i = 0; i < 6; i++) {
      v += a * noise(p);
      p = rot * p * 2.0;
      a *= 0.5;
    }
    return v;
  }

  // -- volumetric cloud density via domain-warped fbm ------------------
  // Two layers of warping -> billowing, organic fog that never repeats.
  float clouds(vec2 p) {
    float t = uTime * 0.06;
    vec2 q = vec2(fbm(p + vec2(0.0, t)), fbm(p + vec2(5.2, 1.3 - t)));
    vec2 r = vec2(
      fbm(p + 3.0 * q + vec2(1.7 - t * 0.5, 9.2)),
      fbm(p + 3.0 * q + vec2(8.3, 2.8 + t * 0.5))
    );
    return fbm(p + 3.5 * r);
  }

  void main() {
    vec2 res = uResolution;
    float aspect = res.x / max(res.y, 1.0);

    // centred, aspect-corrected coords
    vec2 uv = (gl_FragCoord.xy / res) * 2.0 - 1.0;
    uv.x *= aspect;

    // slow drift + optional mouse parallax
    vec2 par = (uMouse - 0.5) * 0.3 * uInteractive;
    vec2 p = uv * 1.15 + vec2(uTime * 0.025, uTime * 0.012) + par;

    // billowing density field
    float d = clouds(p);
    d = smoothstep(0.25, 1.05, d);          // shape the fog
    float d2 = clouds(p * 1.9 + 4.0);        // finer detail layer
    float density = clamp(d * 0.85 + d2 * 0.25, 0.0, 1.0);

    float crest = smoothstep(0.72, 1.0, density);
    float rim = smoothstep(0.30, 0.46, density) * (1.0 - smoothstep(0.46, 0.62, density));
    float vig = 1.0 - dot(uv * vec2(0.40, 0.52), uv * vec2(0.40, 0.52));

    vec3 col;
    if (uLight > 0.5) {
      // -- LIGHT: soft pastel mist that gently TINTS a light base -------
      // build a tint colour, then blend it onto white by density so the
      // page stays bright (no glowing-on-white).
      vec3 tint = mix(uIndigo, uMcp, smoothstep(0.2, 0.85, density));
      tint = mix(tint, uGold, crest * 0.5);
      tint = mix(tint, uInfo, rim * 0.4);
      // pastelise the tint (toward white) so it never gets heavy
      tint = mix(uBase, tint, 0.45);
      float amt = smoothstep(0.05, 0.95, density) * 0.5;
      col = mix(uBase, tint, amt);
      col *= clamp(vig + 0.45, 0.85, 1.0);   // very gentle edge fade on light
    } else {
      // -- DARK / VIOLET: luminous nebula on a near-black base ----------
      col = uBase;
      col = mix(col, uIndigo, smoothstep(0.05, 0.55, density));
      col = mix(col, uMcp,    smoothstep(0.40, 0.92, density) * 0.85);
      col = mix(col, uGold, crest * 0.7);          // gold crests = identity
      col += uInfo * rim * 0.22;                    // cool rim light
      col += uMcp * pow(density, 2.0) * 0.18;       // internal glow
      col *= clamp(vig, 0.55, 1.0);
      col = col / (col + 0.7);                       // filmic tonemap
      col = pow(col, vec3(0.82));
    }

    gl_FragColor = vec4(col, 1.0);
  }
`;

function hexToVec3(hex: string): [number, number, number] {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!m) return [1, 1, 1];
  return [parseInt(m[1], 16) / 255, parseInt(m[2], 16) / 255, parseInt(m[3], 16) / 255];
}

// Per-theme palettes - literal hex mirroring the CSS vars in index.css.
// dark/violet are luminous-on-black; light is a pastel-mist palette.
const PALETTES: Record<FlowTheme, {
  base: string; mcp: string; gold: string; info: string; indigo: string; light: boolean;
}> = {
  dark:   { base: "#0A0A0C", mcp: "#8B5CF6", gold: "#E0B96A", info: "#06B6D4", indigo: "#2E1F5E", light: false },
  violet: { base: "#15121B", mcp: "#A78BFF", gold: "#E0B96A", info: "#3FD0E6", indigo: "#3A2A78", light: false },
  light:  { base: "#F4F2F8", mcp: "#7C5CE0", gold: "#C79A3A", info: "#2BA7C4", indigo: "#9C8AD6", light: true },
};

export function FlowField({
  className = "",
  speed = 0.4,
  opacity = 1,
  interactive = false,
  theme = "dark",
}: FlowFieldProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const reduced =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

    let renderer: Renderer;
    try {
      renderer = new Renderer({
        alpha: false,
        antialias: false,
        dpr: Math.min(window.devicePixelRatio || 1, 2),
      });
    } catch {
      // No WebGL -> leave the container empty; the parent CSS gradient shows.
      return;
    }

    const pal = PALETTES[theme];
    const BASE = hexToVec3(pal.base);

    const gl = renderer.gl;
    gl.clearColor(BASE[0], BASE[1], BASE[2], 1);
    const canvas = gl.canvas as HTMLCanvasElement;
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.display = "block";
    container.appendChild(canvas);

    const geometry = new Triangle(gl);
    const program = new Program(gl, {
      vertex: VERTEX,
      fragment: FRAGMENT,
      uniforms: {
        uResolution: { value: new Vec2(1, 1) },
        uTime: { value: 0 },
        uMouse: { value: new Vec2(0.5, 0.5) },
        uInteractive: { value: interactive ? 1 : 0 },
        uBase: { value: new Vec3(...BASE) },
        uMcp: { value: new Vec3(...hexToVec3(pal.mcp)) },
        uGold: { value: new Vec3(...hexToVec3(pal.gold)) },
        uInfo: { value: new Vec3(...hexToVec3(pal.info)) },
        uIndigo: { value: new Vec3(...hexToVec3(pal.indigo)) },
        uLight: { value: pal.light ? 1 : 0 },
      },
    });
    const mesh = new Mesh(gl, { geometry, program });

    const resize = () => {
      const r = container.getBoundingClientRect();
      const w = Math.max(1, Math.floor(r.width || container.clientWidth || window.innerWidth));
      const h = Math.max(1, Math.floor(r.height || container.clientHeight || window.innerHeight));
      renderer.setSize(w, h);
      program.uniforms.uResolution.value.set(gl.drawingBufferWidth, gl.drawingBufferHeight);
      renderer.render({ scene: mesh });
    };
    const ro = new ResizeObserver(resize);
    ro.observe(container);
    resize();

    const onMouse = (e: MouseEvent) => {
      if (!interactive) return;
      const r = container.getBoundingClientRect();
      program.uniforms.uMouse.value.set(
        (e.clientX - r.left) / r.width,
        1 - (e.clientY - r.top) / r.height,
      );
    };
    if (interactive) window.addEventListener("mousemove", onMouse);

    // Reduced motion -> one static frame, no animation loop.
    if (reduced) {
      program.uniforms.uTime.value = 12.0; // a pleasant frozen pose
      renderer.render({ scene: mesh });
      return () => {
        ro.disconnect();
        if (interactive) window.removeEventListener("mousemove", onMouse);
        try {
          container.removeChild(canvas);
        } catch {
          /* already gone */
        }
      };
    }

    // Pause the loop while offscreen.
    let visible = true;
    const io = new IntersectionObserver(
      ([entry]) => {
        const wasVisible = visible;
        visible = entry.isIntersecting;
        if (visible && !wasVisible) {
          last = performance.now();
          raf = requestAnimationFrame(loop);
        }
      },
      { threshold: 0 },
    );
    io.observe(container);

    let raf = 0;
    let t = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05); // clamp tab-switch jumps
      last = now;
      t += dt * speed;
      program.uniforms.uTime.value = t;
      renderer.render({ scene: mesh });
      if (visible) raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      if (interactive) window.removeEventListener("mousemove", onMouse);
      try {
        container.removeChild(canvas);
      } catch {
        /* already gone */
      }
      const ext = gl.getExtension("WEBGL_lose_context");
      ext?.loseContext();
    };
  }, [speed, interactive, theme]);

  return (
    <div
      ref={containerRef}
      aria-hidden="true"
      className={className}
      style={{ opacity }}
    />
  );
}

export default FlowField;
