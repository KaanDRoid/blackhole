// Styling: colour mapping of the disk, disk texture and background sky.
// Nothing in this file changes where rays go; it only decides what colour a
// ray gets once the geodesic code has classified it.

uniform int u_style;        // 0 = physical, 1 = cyberpunk
uniform float u_time;       // animation loop phase in [0, 1)
uniform float u_pix_angle;  // angular size of one output pixel (rad)
uniform float u_flux_max;   // peak of disk_flux() on [r_in, r_out]
uniform float u_t_peak;     // physical style: emitted temperature (K) at the flux peak
uniform float u_flow;       // disk texture shear per loop, in units of Omega
uniform float u_tex_amp;    // disk texture contrast (0 = smooth disk)
uniform float u_horizon;    // cyberpunk floor horizon elevation (rad)

// ---------------------------------------------------------------- noise

float hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 hash33(vec3 p3) {
    p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

float hash11(float x) {
    return hash13(vec3(x, x * 0.37 + 11.0, x * 1.71 + 5.0));
}

float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(i);
    float n100 = hash13(i + vec3(1, 0, 0));
    float n010 = hash13(i + vec3(0, 1, 0));
    float n110 = hash13(i + vec3(1, 1, 0));
    float n001 = hash13(i + vec3(0, 0, 1));
    float n101 = hash13(i + vec3(1, 0, 1));
    float n011 = hash13(i + vec3(0, 1, 1));
    float n111 = hash13(i + vec3(1, 1, 1));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

float fbm(vec3 p) {
    float s = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; ++i) {
        s += a * vnoise(p);
        p = p * 2.03 + vec3(1.7, 9.2, 3.1);
        a *= 0.5;
    }
    return s;
}

// ---------------------------------------------------------- colour maps

// Approximate sRGB colour of a blackbody (fit valid 1000 K - 40000 K),
// returned in linear RGB with unit maximum.
vec3 blackbody(float t_kelvin) {
    float t = clamp(t_kelvin, 1000.0, 40000.0) / 100.0;
    float r = t <= 66.0 ? 1.0 : clamp(1.292936 * pow(t - 60.0, -0.1332048), 0.0, 1.0);
    float g = t <= 66.0 ? clamp(0.3900816 * log(t) - 0.6318414, 0.0, 1.0)
                        : clamp(1.1298909 * pow(t - 60.0, -0.0755148), 0.0, 1.0);
    float b = t >= 66.0 ? 1.0 : (t <= 19.0 ? 0.0 : clamp(0.5432068 * log(t - 10.0) - 1.1962541, 0.0, 1.0));
    return pow(vec3(r, g, b), vec3(2.2));
}

// Neon palette indexed by the g-shifted relative temperature.
vec3 neon_palette(float x) {
    const vec3 c0 = vec3(0.02, 0.00, 0.06);
    const vec3 c1 = vec3(0.28, 0.02, 0.70);  // violet
    const vec3 c2 = vec3(0.95, 0.04, 0.70);  // magenta
    const vec3 c3 = vec3(1.00, 0.22, 0.50);  // hot pink
    const vec3 c4 = vec3(0.10, 0.85, 1.00);  // cyan
    const vec3 c5 = vec3(1.00, 0.96, 0.62);  // pale yellow
    x = clamp(x, 0.0, 1.6);
    if (x < 0.30) return mix(c0, c1, smoothstep(0.00, 0.30, x));
    if (x < 0.60) return mix(c1, c2, smoothstep(0.30, 0.60, x));
    if (x < 0.85) return mix(c2, c3, smoothstep(0.60, 0.85, x));
    if (x < 1.10) return mix(c3, c4, smoothstep(0.85, 1.10, x));
    return mix(c4, c5, smoothstep(1.10, 1.45, x));
}

// Amber phosphor (P3-like) ramp for the terminal style: dim brown, the
// #ffb000 amber of the profile terminal, then a pale overdriven core.
const vec3 AMBER = vec3(1.0, 0.434, 0.0);   // #ffb000 in linear RGB
const vec3 CYAN = vec3(0.05, 0.89, 1.0);    // #3ff2ff in linear RGB

vec3 phosphor(float v) {
    const vec3 dim = vec3(0.28, 0.10, 0.0);
    const vec3 hot = vec3(1.0, 0.80, 0.45);
    return v < 1.0 ? mix(dim, AMBER, smoothstep(0.0, 1.0, v))
                   : mix(AMBER, hot, smoothstep(1.0, 2.2, v));
}

// ---------------------------------------------------------- disk texture

float disk_layer(float r, float phi, float seed) {
    vec3 p = vec3(1.6 * cos(phi), 1.6 * sin(phi), 14.0 * log(r) + seed);
    float n = smoothstep(0.2, 0.8, fbm(p));
    float lanes = 0.5 + 0.5 * sin(48.0 * log(r) + 7.0 * n);
    return mix(0.15, 1.6, n) * mix(0.6, 1.2, lanes);
}

// Two copies of the pattern are sheared by the Keplerian angular velocity and
// cross-faded, so the differential rotation repeats exactly after one loop.
float disk_texture(float r, float phi) {
    float omega = sqrt(0.5 / (r * r * r));
    float t0 = fract(u_time);
    float t1 = fract(u_time + 0.5);
    float a = disk_layer(r, phi - u_flow * omega * t0, 0.0);
    float b = disk_layer(r, phi - u_flow * omega * t1, 37.0);
    float w = pow(sin(PI * t0), 2.0);
    float tex = mix(b, a, w);
    return mix(1.0, tex, u_tex_amp);
}

vec3 disk_color(Hit h) {
    float f = max(disk_flux(h.r), 0.0) / u_flux_max;  // relative emitted flux
    float t_rel = pow(f, 0.25);                       // relative emitted temperature
    float g = h.g;
    float lum = pow(g, 4.0) * f;                      // bolometric I_obs = g^4 I_emit
    float tex = disk_texture(h.r, h.phi_disk);
    if (u_style == 0) {
        return blackbody(u_t_peak * g * t_rel) * lum * tex;
    }
    if (u_style == 2) {
        // terminal: single amber phosphor, brightness carries g^4 F; images
        // after the first equatorial crossing are drawn in the cyan accent
        float v = pow(lum, 0.55) * tex;
        vec3 ink = h.order >= 1 ? mix(phosphor(v), CYAN, 0.75) * 0.6 : phosphor(v);
        return ink * v * 1.4;
    }
    // cyberpunk: palette instead of a blackbody, compressed dynamic range
    return neon_palette(g * t_rel) * pow(lum, 0.6) * tex * 1.6;
}

// ------------------------------------------------------------ background

vec3 star_layer(vec3 d, float scale, float density, float seed) {
    vec3 p = d * scale;
    vec3 c = floor(p);
    vec3 h = hash33(c + seed);
    if (h.x > density) return vec3(0.0);
    vec3 sp = normalize(c + 0.2 + 0.6 * hash33(c + seed + 17.0)) * scale;
    float dist = length(p - sp);
    float sigma = max(0.7 * u_pix_angle * scale, 0.02);
    float bright = 0.15 + 2.5 * pow(h.y, 6.0);
    vec3 tint;
    if (u_style == 0) {
        tint = h.z < 0.3 ? vec3(1.0, 0.8, 0.6) : (h.z < 0.6 ? vec3(0.75, 0.85, 1.0) : vec3(1.0));
    } else {
        tint = h.z < 0.35 ? vec3(0.3, 0.9, 1.0) : (h.z < 0.65 ? vec3(1.0, 0.4, 0.9) : vec3(0.95, 0.9, 1.0));
    }
    return tint * bright * exp(-0.5 * dist * dist / (sigma * sigma));
}

vec3 starfield(vec3 d) {
    return star_layer(d, 60.0, 0.10, 0.0) + star_layer(d, 150.0, 0.06, 51.0)
         + 0.5 * star_layer(d, 320.0, 0.04, 97.0);
}

vec3 sky_physical(vec3 d) {
    float band = exp(-pow(d.z * 4.0 - 0.6 * d.x, 2.0));
    vec3 glow = vec3(0.010, 0.009, 0.012) * band * (0.5 + fbm(d * 4.0));
    return starfield(d) + glow;
}

// Building silhouettes around the horizon. Returns coverage in .w.
vec4 skyline(float el, float az, float scale, float max_h, float seed, vec3 window_tint) {
    float x = (az / (2.0 * PI) + 0.5) * scale;
    float id = floor(x);
    float hb = 0.008 + max_h * pow(hash11(id + seed), 2.0);
    if (hash11(id + seed + 7.0) > 0.92) hb += max_h * 0.9 * hash11(id + seed + 3.0);
    if (el > hb || el < -0.02) return vec4(0.0);
    vec3 col = vec3(0.004, 0.0, 0.012);
    // lit windows
    float fx = fract(x);
    if (fx > 0.12 && fx < 0.88) {
        vec2 cell = vec2(floor(fx * 5.0), floor(el * 520.0));
        vec2 inner = vec2(fract(fx * 5.0), fract(el * 520.0));
        float on = step(0.78, hash13(vec3(cell, id + seed)));
        float shape = step(0.25, inner.x) * step(inner.x, 0.75) * step(0.3, inner.y) * step(inner.y, 0.7);
        vec3 tint = hash13(vec3(cell.yx, id)) < 0.5 ? window_tint : vec3(1.0, 0.85, 0.45);
        col += tint * on * shape * 0.5;
    }
    // neon rim on the roof line
    col += window_tint * 2.0 * smoothstep(0.0025, 0.0, hb - el);
    return vec4(col, 1.0);
}

vec3 sky_cyberpunk(vec3 d) {
    float el = asin(clamp(d.z, -1.0, 1.0)) - u_horizon;
    float az = atan(d.y, d.x);
    vec3 col = vec3(0.012, 0.004, 0.03);
    col += vec3(0.85, 0.05, 0.55) * 0.12 * exp(-abs(el) / 0.05);
    col += vec3(0.25, 0.02, 0.60) * 0.08 * exp(-abs(el) / 0.30);

    if (el >= 0.0) {
        float neb = fbm(d * 3.0 + vec3(4.0, 1.0, 7.0));
        col += mix(vec3(0.20, 0.01, 0.35), vec3(0.0, 0.25, 0.35), vnoise(d * 2.0)) * 0.10 * pow(neb, 3.0) * 4.0;
        col += starfield(d) * smoothstep(0.0, 0.12, el);
        vec4 far = skyline(el, az, 340.0, 0.035, 0.0, vec3(0.2, 0.8, 1.0));
        col = mix(col, far.rgb + col * 0.25, far.w);
        vec4 near = skyline(el, az, 150.0, 0.07, 113.0, vec3(1.0, 0.15, 0.7));
        col = mix(col, near.rgb, near.w);
        return col;
    }

    // perspective grid on a floor plane below the horizon
    float depth = sin(-el);
    vec2 q = vec2(cos(az), sin(az)) * cos(-el) / depth * 1.2;
    float w = 1.2 * u_pix_angle / (depth * depth);  // grid units per pixel
    vec2 f = abs(fract(q) - 0.5);
    float dl = 0.5 - max(f.x, f.y);
    const float lw = 0.025;
    float line = 1.0 - smoothstep(lw, lw + max(w, 1e-4), dl);
    float resolved = smoothstep(0.6, 0.08, w);
    line = mix(4.0 * lw, line, resolved);
    vec3 grid_col = mix(vec3(0.1, 0.85, 1.0), vec3(1.0, 0.1, 0.75), smoothstep(2.0, 25.0, length(q)));
    col = vec3(0.006, 0.0, 0.02) + col * 0.6;
    col += grid_col * line * 0.45 * exp(-length(q) * 0.03);
    return col;
}

// Terminal sky: a coordinate graticule on the celestial sphere, drawn like a
// vector display, plus sparse dim stars. Lensing bends the grid lines, so the
// background doubles as a map of where escaped rays came from.
vec3 sky_terminal(vec3 d) {
    float lat = asin(clamp(d.z, -1.0, 1.0));
    float lon = atan(d.y, d.x);
    const float step_deg = 10.0;
    float s = radians(step_deg);
    float w = 1.4 * u_pix_angle;                       // line half-width, rad
    float dlat = abs(fract(lat / s + 0.5) - 0.5) * s;
    float dlon = abs(fract(lon / s + 0.5) - 0.5) * s * max(cos(lat), 0.05);
    float line = max(1.0 - smoothstep(0.0, w, dlat), 1.0 - smoothstep(0.0, w, dlon));
    float equator = 1.0 - smoothstep(0.0, 1.6 * w, abs(lat));
    vec3 col = vec3(0.004, 0.0015, 0.0);
    col += AMBER * 0.10 * line + AMBER * 0.22 * equator;
    float lum = dot(starfield(d), vec3(0.3333));
    col += AMBER * 0.5 * lum;
    return col;
}

vec3 background(vec3 d) {
    if (u_style == 2) return sky_terminal(d);
    return u_style == 0 ? sky_physical(d) : sky_cyberpunk(d);
}
