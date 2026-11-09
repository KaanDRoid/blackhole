// Physics: Schwarzschild null geodesics, thin disk crossings, redshift.
// Port of blackhole/physics.py (same algorithm, single precision).
// Units: G = c = 1, Schwarzschild radius r_s = 1, disk in the z = 0 plane.

const float PI = 3.14159265358979;

const int ESCAPED = 0;
const int CAPTURED = 1;
const int DISK = 2;
const int UNRESOLVED = 3;

uniform float u_r_obs;     // observer radius
uniform float u_r_in;      // inner disk edge (ISCO = 3)
uniform float u_r_out;     // outer disk edge
uniform int u_disk_on;     // 0 disables the disk (shadow measurement)
uniform int u_max_steps;
uniform float u_h_max;     // max step in phi
uniform float u_du_max;    // max relative change of u per step

struct Hit {
    int status;
    int order;      // index of the equatorial crossing that hit the disk
    float r;        // emission radius
    float phi_disk; // azimuth of the emission point
    float g;        // nu_obs / nu_emit
    vec3 dir;       // asymptotic direction of an escaped ray
};

// Binet equation u'' = -u + 1.5 u^2, state = (u, du/dphi)
vec2 binet(vec2 s) {
    return vec2(s.y, -s.x + 1.5 * s.x * s.x);
}

vec2 rk4(vec2 s, float h) {
    vec2 k1 = binet(s);
    vec2 k2 = binet(s + 0.5 * h * k1);
    vec2 k3 = binet(s + 0.5 * h * k2);
    vec2 k4 = binet(s + h * k3);
    return s + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

float hermite(vec2 s0, vec2 s1, float h, float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * s0.x + (t3 - 2.0 * t2 + t) * h * s0.y
         + (-2.0 * t3 + 3.0 * t2) * s1.x + (t3 - t2) * h * s1.y;
}

// Page-Thorne flux for a = 0 (arbitrary normalisation), zero at the ISCO.
float disk_flux(float r) {
    if (r <= 3.0) return 0.0;
    float x = sqrt(2.0 * r);  // sqrt(r / M)
    const float s3 = 1.7320508;
    const float x0 = 2.4494897;
    float l = log((x + s3) * (x0 - s3) / ((x - s3) * (x0 + s3)));
    return (x - x0 + 0.5 * s3 * l) / ((x * x - 3.0) * pow(x, 5.0));
}

// Gravitational + Doppler shift for gas on a Keplerian circular orbit,
// photon with E = 1 and axial angular momentum lz, static observer.
float redshift(float r, float lz) {
    float omega = sqrt(0.5 / (r * r * r));
    return sqrt(1.0 - 1.5 / r) / ((1.0 - omega * lz) * sqrt(1.0 - 1.0 / u_r_obs));
}

Hit trace(vec3 pos, vec3 d) {
    Hit hit;
    hit.status = UNRESOLVED;
    hit.order = -1;
    hit.r = 0.0;
    hit.phi_disk = 0.0;
    hit.g = 0.0;
    hit.dir = vec3(0.0);

    float r0 = length(pos);
    vec3 e1 = pos / r0;
    float cosa = dot(d, e1);
    vec3 perp = d - cosa * e1;
    float sina = length(perp);
    if (sina < 1e-7) {
        vec3 helper = abs(e1.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
        perp = cross(e1, helper);
        sina = 1e-7;
    }
    vec3 e2 = normalize(perp);

    float u0 = 1.0 / r0;
    vec2 s = vec2(u0, -u0 * sqrt(1.0 - u0) * cosa / sina);
    float b = r0 * sina / sqrt(1.0 - u0);
    float lz = -b * cross(e1, e2).z;

    float A = e1.z;
    float B = e2.z;
    bool planar = (A * A + B * B) < 1e-12;
    float phi_next = mod(atan(B, A) + 0.5 * PI, PI);
    if (phi_next <= 1e-6) phi_next += PI;
    if (planar) phi_next = 1e30;

    float phi = 0.0;
    int crossings = 0;
    for (int i = 0; i < u_max_steps; ++i) {
        float h = min(u_h_max, u_du_max * max(s.x, 0.05) / (abs(s.y) + 1e-9));
        vec2 s1 = rk4(s, h);
        float phi1 = phi + h;

        if (phi1 >= phi_next) {
            float uc = hermite(s, s1, h, (phi_next - phi) / h);
            if (u_disk_on == 1 && uc > 0.0 && uc < 1.0) {
                float rc = 1.0 / uc;
                if (rc >= u_r_in && rc <= u_r_out) {
                    vec3 p = rc * (cos(phi_next) * e1 + sin(phi_next) * e2);
                    hit.status = DISK;
                    hit.order = crossings;
                    hit.r = rc;
                    hit.phi_disk = atan(p.y, p.x);
                    hit.g = redshift(rc, lz);
                    return hit;
                }
            }
            crossings += 1;
            phi_next += PI;
        }
        if (s1.x >= 1.0) {
            hit.status = CAPTURED;
            return hit;
        }
        if (s1.x <= 0.0) {
            float phi_inf = phi + h * s.x / (s.x - s1.x);
            hit.status = ESCAPED;
            hit.dir = cos(phi_inf) * e1 + sin(phi_inf) * e2;
            return hit;
        }
        s = s1;
        phi = phi1;
    }
    return hit;
}
