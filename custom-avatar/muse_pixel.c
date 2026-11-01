// Copyright (c) Meta Platforms, Inc. and affiliates.

/*
 * The avatar: a glossy black disc, like a badge, with a HAL-style red eye
 * at its centre -- concentric rings of red around a small bright orange
 * core that slowly swells and dims as if breathing -- and a white voice
 * waveform, a sine wave, laid over the eye on its own layer. The wave is
 * the face's expression: it ripples gently when idle, grows with the
 * voice when speaking or listening, flattens as the gadget powers down.
 * No limbs, no mouth; the eye's glow and the wave carry every state.
 * Personality: calm, watchful, a little uncanny -- a quiet machine eye
 * that listens.
 *
 * Touching the avatar (the pet reaction, pose->happy) jumps the wave
 * to a random one of its five shapes -- sine, sawtooth, triangle, square
 * and a sharpened pulse sine -- never the shape already showing, and the
 * new shape stays until the next touch. ERROR always overrides the wave
 * with its jagged square wave.
 */

#include "muse_pixel.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define W MUSE_PX_W
#define H MUSE_PX_H
#define TAU 6.2831853f

/* ---------------------------------------------------------------------------
 * Palette
 * ------------------------------------------------------------------------- */

enum {
    C_BG = 0,
    C_OUT,       /* outline */
    C_OUT2,      /* soft outline where the hood tucks around the face */
    C_BD,        /* disc dark */
    C_BM,        /* disc mid */
    C_BL,        /* disc light */
    C_BH,        /* disc highlight / gloss */
    C_RIM,       /* state-tinted rim light */
    C_SKIND,     /* eye deep red */
    C_SKIN,
    C_SKINL,
    C_IRIS,      /* bead eyes */
    C_SHINE,
    C_BROW,
    C_BLUSH,
    C_BLUSHD,
    C_MOUTH,
    C_TONGUE,
    C_G0,        /* state glow ramp, bright ... */
    C_G1,
    C_G2,
    C_G3,        /* ... deep */
    C_AURA1,
    C_AURA2,
    C_SPK,
    C_ACC,
    C_SHADOW,
    C_HEART,
    C_WHITE,
    C_COUNT,
};

typedef struct {
    float r, g, b;
} rgb_t;

/* Per-mode glow ramp (bright -> deep) and accent. */
typedef struct {
    uint32_t f[4];
    uint32_t acc;
} scheme_t;

static const scheme_t SCHEMES[MUSE_MODE_COUNT] = {
    [MUSE_MODE_BOOT]      = { { 0xffffff, 0xcfe0ff, 0x8fa8ff, 0x5a5fe0 }, 0xa9c0ff },
    [MUSE_MODE_IDLE]      = { { 0xf4e8ff, 0xc7a4ff, 0x9a6bff, 0x5b3fd9 }, 0xa77dff },
    [MUSE_MODE_LISTENING] = { { 0xe8faff, 0x8fdcff, 0x3fa2ff, 0x2a5bd7 }, 0x5cb8ff },
    [MUSE_MODE_THINKING]  = { { 0xffe6ff, 0xff9cf0, 0xd35bff, 0x7a2bd9 }, 0xe07bff },
    [MUSE_MODE_SPEAKING]  = { { 0xeafff4, 0x9ff5cf, 0x3fd9a0, 0x1f9a7a }, 0x6ff0bf },
    [MUSE_MODE_ERROR]     = { { 0xffd6d6, 0xff6b6b, 0xc7304a, 0x6b1a3a }, 0xff5c5c },
    [MUSE_MODE_OFF]       = { { 0xd8d4ff, 0x8f86d9, 0x5a4fb0, 0x2e2870 }, 0x7c72d0 },
};

/* Glossy black disc and a red HAL eye (SKIN tones are the eye reds). */
static const uint32_t FIXED[C_COUNT] = {
    [C_BG] = 0x000000,
    [C_OUT] = 0x060607,
    [C_OUT2] = 0x2a0208,
    [C_BD] = 0x0b0b0e,
    [C_BM] = 0x17171c,
    [C_BL] = 0x232329,
    [C_BH] = 0x44444f,
    [C_SKIND] = 0x5f0a06,
    [C_SKIN] = 0xa8160d,
    [C_SKINL] = 0xe8321e,
    [C_IRIS] = 0xffa63d,
    [C_SHINE] = 0xffffff,
    [C_BROW] = 0x38060a,
    [C_BLUSH] = 0x8f1d12,
    [C_BLUSHD] = 0x6e0f08,
    [C_MOUTH] = 0x12030a,
    [C_TONGUE] = 0xff6a3d,
    [C_SHADOW] = 0x16101f,
    [C_HEART] = 0xff4f8b,
    [C_WHITE] = 0xffffff,
};

static rgb_t s_scheme[5];      /* live, blended: f0..f3, acc */
static bool s_scheme_init;
static uint16_t s_pal[C_COUNT];
static uint16_t s_pal_dim[C_COUNT];

static uint8_t s_fb[W * H];
static uint8_t s_mask[W * H];

static const uint8_t BAYER4[4][4] = {
    { 0, 8, 2, 10 },
    { 12, 4, 14, 6 },
    { 3, 11, 1, 9 },
    { 15, 7, 13, 5 },
};

/*
 * The per-pixel work is fixed point (Q12: ONE = 1.0), with tables for the
 * powers and roots: chips without an FPU (ESP32-C6) emulate float in
 * software, which made a frame take 250 ms. At 64 px the quantisation is
 * invisible.
 */
#define Q 12
#define ONE (1 << Q)
#define QF(v) ((int32_t)((v) * ONE))
#define POW_LUT_N 256
#define POW_LUT_MAX_Q QF(1.2f)          /* body_field() bails out beyond this */
#define SQRT_LUT_N 1024                 /* indexed by x >> 2 */
static int16_t s_pow_dome[POW_LUT_N + 1];   /* |u|^2.7 */
static int16_t s_pow_base[POW_LUT_N + 1];   /* |u|^3.6 */
static int16_t s_sqrt[SQRT_LUT_N + 1];

static void init_luts(void)
{
    for (int i = 0; i <= POW_LUT_N; i++) {
        float u = (float)i * POW_LUT_MAX_Q / POW_LUT_N / ONE;
        s_pow_dome[i] = (int16_t)(powf(u, 2.7f) * ONE);
        s_pow_base[i] = (int16_t)(powf(u, 3.6f) * ONE);
    }
    for (int i = 0; i <= SQRT_LUT_N; i++) {
        s_sqrt[i] = (int16_t)(sqrtf((float)i / SQRT_LUT_N) * ONE);
    }
}

/* a in [0, POW_LUT_MAX_Q] */
static inline int32_t pow_q(const int16_t *lut, int32_t a)
{
    return lut[(a * (POW_LUT_N * 65536 / POW_LUT_MAX_Q)) >> 16];
}

/* sqrt of x in [0, ONE] */
static inline int32_t sqrt_q(int32_t x)
{
    return s_sqrt[x >> 2];
}

static inline int32_t bayer_q(int x, int y)
{
    return BAYER4[y & 3][x & 3] * (ONE / 16) + ONE / 32;
}

static inline float bayer(int x, int y)
{
    return (BAYER4[y & 3][x & 3] + 0.5f) / 16.0f;
}

static inline rgb_t hex_rgb(uint32_t c)
{
    return (rgb_t){ (float)((c >> 16) & 0xff), (float)((c >> 8) & 0xff), (float)(c & 0xff) };
}

static inline rgb_t mix(rgb_t a, rgb_t b, float t)
{
    return (rgb_t){ a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t };
}

static inline rgb_t scale_rgb(rgb_t a, float k)
{
    return (rgb_t){ a.r * k, a.g * k, a.b * k };
}

static inline uint16_t to565(rgb_t c)
{
    int r = (int)(c.r + 0.5f), g = (int)(c.g + 0.5f), b = (int)(c.b + 0.5f);
    r = r < 0 ? 0 : (r > 255 ? 255 : r);
    g = g < 0 ? 0 : (g > 255 ? 255 : g);
    b = b < 0 ? 0 : (b > 255 ? 255 : b);
    return (uint16_t)(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
}

uint32_t muse_pixel_accent(muse_mode_t mode)
{
    return SCHEMES[mode < MUSE_MODE_COUNT ? mode : MUSE_MODE_IDLE].acc;
}

static void update_palette(const scheme_t *target, float dt)
{
    rgb_t tgt[5];
    for (int i = 0; i < 4; i++) {
        tgt[i] = hex_rgb(target->f[i]);
    }
    tgt[4] = hex_rgb(target->acc);

    float k = s_scheme_init ? 1.0f - expf(-dt * 7.0f) : 1.0f;
    for (int i = 0; i < 5; i++) {
        s_scheme[i] = mix(s_scheme[i], tgt[i], k);
    }
    s_scheme_init = true;

    rgb_t pal[C_COUNT];
    for (int i = 0; i < C_COUNT; i++) {
        pal[i] = hex_rgb(FIXED[i]);
    }
    rgb_t acc = s_scheme[4];
    pal[C_G0] = s_scheme[0];
    pal[C_G1] = s_scheme[1];
    pal[C_G2] = s_scheme[2];
    pal[C_G3] = s_scheme[3];
    pal[C_ACC] = acc;
    pal[C_RIM] = mix(pal[C_BL], acc, 0.45f);
    pal[C_AURA1] = scale_rgb(acc, 0.16f);
    pal[C_AURA2] = scale_rgb(acc, 0.34f);
    pal[C_SPK] = mix(acc, pal[C_WHITE], 0.45f);

    for (int i = 0; i < C_COUNT; i++) {
        s_pal[i] = to565(pal[i]);
        /* The block edge shade gives the enlarged pixels a faint grid texture. */
        s_pal_dim[i] = to565(scale_rgb(pal[i], 0.72f));
    }
}

/* ---------------------------------------------------------------------------
 * Primitive helpers
 * ------------------------------------------------------------------------- */

static inline void px(int x, int y, uint8_t c)
{
    if ((unsigned)x < W && (unsigned)y < H) {
        s_fb[y * W + x] = c;
    }
}

static inline uint8_t get_px(int x, int y)
{
    if ((unsigned)x < W && (unsigned)y < H) {
        return s_fb[y * W + x];
    }
    return C_BG;
}

static inline int iround(float v)
{
    return (int)floorf(v + 0.5f);
}

static inline float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float fracf(float v)
{
    return v - floorf(v);
}

/* Cheap deterministic pseudo-random for idle behaviour. */
static uint32_t s_rng = 0x9e3779b9u;
static float frand(void)
{
    s_rng ^= s_rng << 13;
    s_rng ^= s_rng >> 17;
    s_rng ^= s_rng << 5;
    return (float)(s_rng & 0xffffff) / (float)0x1000000;
}

/* Draw a small bitmap given as rows of '.'/'#'/'o' characters. */
static void stamp(const char *const *rows, int nrows, int x0, int y0, uint8_t fill, uint8_t alt)
{
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; rows[r][c]; c++) {
            char ch = rows[r][c];
            if (ch == '#') {
                px(x0 + c, y0 + r, fill);
            } else if (ch == 'o') {
                px(x0 + c, y0 + r, alt);
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * Idle behaviour: blinking and gaze
 * ------------------------------------------------------------------------- */

typedef struct {
    float next_blink;
    float blink_start;
    float next_gaze;
    float gx, gy;          /* current gaze, -1..1 */
    float tgx, tgy;        /* target gaze */
    float last_t;
} eyes_t;

static eyes_t s_eyes = { .next_blink = 1.5f, .blink_start = -10, .next_gaze = 1.0f };

static float eyes_update(const muse_pose_t *p, float dt)
{
    eyes_t *e = &s_eyes;

    if (p->t >= e->next_blink) {
        e->blink_start = p->t;
        /* Occasionally double-blink. */
        e->next_blink = p->t + (frand() < 0.2f ? 0.28f : 2.2f + frand() * 3.0f);
    }

    if (p->t >= e->next_gaze) {
        e->next_gaze = p->t + 1.2f + frand() * 2.4f;
        if (frand() < 0.35f) {
            e->tgx = 0;
            e->tgy = 0;
        } else {
            e->tgx = frand() * 2 - 1;
            e->tgy = (frand() * 2 - 1) * 0.6f;
        }
    }

    float tgx = e->tgx, tgy = e->tgy;
    switch (p->mode) {
    case MUSE_MODE_LISTENING:
        tgx = 0;
        tgy = 0.1f;
        break;
    case MUSE_MODE_THINKING:
        tgx = 0.75f * sinf(p->mode_t * 1.3f) + 0.25f;
        tgy = -0.85f;
        break;
    case MUSE_MODE_SPEAKING:
        tgx *= 0.3f;
        tgy = 0;
        break;
    default:
        break;
    }
    float k = 1.0f - expf(-dt * 14.0f);
    e->gx += (tgx - e->gx) * k;
    e->gy += (tgy - e->gy) * k;

    /* Blink curve: 0 = open, 1 = shut. */
    float bt = (p->t - e->blink_start) / 0.16f;
    if (bt < 0 || bt > 1) {
        return 0;
    }
    return 1.0f - fabsf(bt * 2 - 1);
}


/* ---------------------------------------------------------------------------
 * Scene layers
 * ------------------------------------------------------------------------- */

static void draw_aura(float cx, float cy, float radius, float strength)
{
    int x0 = (int)(cx - radius - 1), x1 = (int)(cx + radius + 1);
    int y0 = (int)(cy - radius - 1), y1 = (int)(cy + radius + 1);
    /* Distances in 1/16 px; the root of d2 / r2 comes from the table. */
    int32_t cx16 = (int32_t)(cx * 16), r2 = (int32_t)(radius * radius * 256);
    int32_t to_idx = (int32_t)((float)SQRT_LUT_N * 65536 / r2);
    int32_t str = QF(strength);
    for (int y = y0; y <= y1; y++) {
        int32_t dy = (int32_t)((y + 0.5f - cy) * 1.1f * 16);
        int32_t dy2 = dy * dy;
        if (dy2 >= r2) {
            continue;
        }
        for (int x = x0; x <= x1; x++) {
            int32_t dx = x * 16 + 8 - cx16;
            int32_t d2 = dx * dx + dy2;
            if (d2 >= r2) {
                continue;
            }
            int32_t i = ((ONE - s_sqrt[(d2 * to_idx) >> 16]) * str) >> Q;
            int32_t b = bayer_q(x, y);
            if (i > QF(0.55f) + ((b * QF(0.35f)) >> Q)) {
                px(x, y, C_AURA2);
            } else if (i > (b * QF(0.9f)) >> Q) {
                px(x, y, C_AURA1);
            }
        }
    }
}

/* Expanding dotted rings (listening / speaking). */
static void draw_rings(float cx, float cy, float t, float level, float speed)
{
    for (int k = 0; k < 2; k++) {
        float ph = fracf(t * speed + k * 0.5f);
        float r = 20 + ph * 11;
        float fade = (1 - ph) * (0.35f + level);
        int n = (int)(r * 2.2f);
        for (int i = 0; i < n; i++) {
            float a = i * TAU / n;
            int x = iround(cx + cosf(a) * r);
            int y = iround(cy + sinf(a) * r * 0.92f);
            if (get_px(x, y) == C_BG || get_px(x, y) == C_AURA1) {
                if (bayer(x, y) < fade) {
                    px(x, y, fade > 0.6f ? C_ACC : C_AURA2);
                }
            }
        }
    }
}

static void draw_shadow(float cx, float y, float half_w)
{
    for (int row = 0; row < 3; row++) {
        float hw = half_w * (row == 1 ? 1.0f : 0.72f);
        for (int x = iround(cx - hw); x <= iround(cx + hw); x++) {
            float edge = fabsf(x + 0.5f - cx) / hw;
            if (bayer(x, (int)y + row) > edge * 0.8f) {
                px(x, (int)y + row, C_SHADOW);
            }
        }
    }
}

/* Stable per-position hash (0..65535) so fur tufts don't shimmer as the avatar moves. */
static inline int32_t hash16(int x, int y)
{
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return (int32_t)((h ^ (h >> 16)) & 0xffff);
}


/*
 * The disc and its eye, in Q12 fixed point like the rest of the renderer.
 * The disc is shaded from a fake surface normal (top-left light) with a
 * glossy sheen band near the top edge; the eye is concentric: dark ring
 * separators, red bands that brighten toward the middle, and the small
 * bright core. eye_gain (per frame) is the pulse: it swells and dims the
 * whole eye, and a blink just turns the gain down.
 */
static void draw_disc(float cx, float cy, float rx, float ry,
                      float ex, float ey, float eye_r, float eye_gain, float fade)
{
    int32_t cx16 = (int32_t)(cx * 16), cy16 = (int32_t)(cy * 16);
    int32_t ex16 = (int32_t)(ex * 16), ey16 = (int32_t)(ey * 16);
    int32_t rx16 = (int32_t)(rx * 16), ry16 = (int32_t)(ry * 16);
    int32_t er16 = (int32_t)(eye_r * 16);
    int32_t gain_q = QF(eye_gain);
    int32_t fade_q = QF(fade);
    int x0 = (int)(cx - rx - 1), x1 = (int)(cx + rx + 1);
    int y0 = (int)(cy - ry - 1), y1 = (int)(cy + ry + 1);
    for (int y = y0; y <= y1; y++) {
        int32_t dy16 = y * 16 + 8 - cy16;
        int32_t ny = dy16 * ONE / ry16;
        int32_t edy16 = y * 16 + 8 - ey16;
        for (int x = x0; x <= x1; x++) {
            int32_t dx16 = x * 16 + 8 - cx16;
            int32_t nx = dx16 * ONE / rx16;
            int32_t r2 = (nx * nx + ny * ny) >> Q;
            if (r2 > ONE) {
                continue;
            }
            s_mask[y * W + x] = 1;
            int32_t nz = sqrt_q(ONE - r2);
            int32_t l = (-QF(0.40f) * nx - QF(0.50f) * ny + QF(0.76f) * nz) >> Q;
            l -= ((ONE - fade_q) * QF(0.9f)) >> Q;
            int32_t b = bayer_q(x, y);
            int32_t lv = l + (((b - ONE / 2) * QF(0.2f)) >> Q);
            uint8_t c = lv > QF(1.12f) ? C_BH : lv > QF(0.94f) ? C_BL : lv > QF(0.55f) ? C_BM : C_BD;
            /* Glossy sheen: a thin arc near the top edge of the disc. */
            if (ny < -QF(0.55f) && r2 > QF(0.76f) && r2 < QF(0.93f) && nx < QF(0.45f)) {
                c = C_BL;
            }
            /* Edge: state-tinted rim on the lit side, near-black elsewhere. */
            if (r2 > QF(0.90f)) {
                int32_t lit = (-nx * QF(0.6f) - ny * QF(0.8f)) >> Q;
                c = (lit > QF(0.55f) && b < QF(0.45f)) ? C_RIM : C_OUT;
            }
            /* The eye, over the disc shading. */
            int32_t edx16 = x * 16 + 8 - ex16;
            int32_t e2 = edx16 * edx16 + edy16 * edy16;
            int32_t er2 = er16 * er16;
            if (e2 <= er2) {
                int32_t q = (int32_t)(((int64_t)e2 << Q) / er2);
                int32_t rr = sqrt_q(q);
                int32_t band5 = rr * 5;
                int band = (int)(band5 >> Q);
                int32_t fr = band5 - (band << Q);
                if (rr < QF(0.22f)) {
                    c = eye_gain < 0.35f ? C_SKIND : C_IRIS;
                } else if (fr < QF(0.08f)) {
                    c = C_BROW;   /* the dark line between two rings */
                } else {
                    int32_t bright = ((ONE - rr) * gain_q) >> Q;
                    bright += (((b - ONE / 2) * QF(0.25f)) >> Q);
                    bright += (((hash16(x, y) >> 5) - ONE / 2) * QF(0.03f)) >> Q;
                    if (band & 1) {
                        bright -= QF(0.10f);
                    }
                    c = bright > QF(0.66f) ? C_SKINL : bright > QF(0.36f) ? C_SKIN : C_SKIND;
                }
            }
            s_fb[y * W + x] = c;
        }
    }
}

enum {
    WAVE_SINE = 0,
    WAVE_SAWTOOTH,
    WAVE_TRIANGLE,
    WAVE_SQUARE,
    WAVE_PULSE,
    WAVE_COUNT
};

/* The wave shape chosen by touching the avatar; persists across modes. */
static int s_wave_pattern = WAVE_SINE;
static float s_prev_happy;

/*
 * The voice waveform across the middle of the disc, in the shape picked
 * by touching the avatar (see s_wave_pattern). Drawn only where the disc
 * is (the part mask), two pixels thick when it swings wide. In ERROR the
 * caller passes WAVE_SQUARE and the alert colour.
 */
static void draw_waveform(float cx, float cy, float half_w, float amp,
                          float phase, int pattern, float fade, uint8_t colour)
{
    int n = (int)(half_w * 2);
    for (int i = 0; i <= n; i++) {
        int x = iround(cx - half_w) + i;
        float u = (x - cx) * 0.40f;
        float a = phase + u;
        float s;
        switch (pattern) {
        case WAVE_SAWTOOTH:
            s = 2.0f * fracf(a / TAU + 0.25f) - 1.0f;
            break;
        case WAVE_TRIANGLE: {
            float f = fracf(a / TAU);
            s = f < 0.5f ? 4.0f * f - 1.0f : 3.0f - 4.0f * f;
            break;
        }
        case WAVE_SQUARE:
            s = sinf(a) >= 0 ? 0.78f : -0.78f;
            break;
        case WAVE_PULSE: {
            float sn = sinf(a);
            s = sn * fabsf(sn);
            break;
        }
        default:   /* WAVE_SINE, with a touch of second harmonic */
            s = 0.78f * sinf(a) + 0.22f * sinf(u * 2.3f + phase + 1.3f);
            break;
        }
        int y = iround(cy + s * amp);
        if (fade < 1.0f && bayer(x, y) > fade) {
            continue;
        }
        if ((unsigned)x < W && (unsigned)y < H && s_mask[y * W + x]) {
            px(x, y, colour);
            if (amp > 4.0f && y + 1 < H && s_mask[(y + 1) * W + x]) {
                px(x, y + 1, colour);
            }
        }
    }
}

static void draw_sparkle(int x, int y, float twinkle, bool front)
{
    uint8_t arm = front ? C_ACC : C_AURA2;
    uint8_t core = front ? C_WHITE : C_SPK;
    if (twinkle > 0.8f) {
        px(x, y, core);
        for (int k = 1; k <= 2; k++) {
            uint8_t c = k == 1 ? (front ? C_SPK : arm) : arm;
            px(x + k, y, c);
            px(x - k, y, c);
            px(x, y + k, c);
            px(x, y - k, c);
        }
    } else if (twinkle > 0.45f) {
        px(x, y, front ? C_SPK : arm);
        px(x + 1, y, arm);
        px(x - 1, y, arm);
        px(x, y + 1, arm);
        px(x, y - 1, arm);
    } else if (twinkle > 0.15f) {
        px(x, y, arm);
    }
}

static void draw_sparkles(const muse_pose_t *p, float cx, float cy, bool front, float speed, int count)
{
    for (int i = 0; i < count; i++) {
        float a = p->t * speed + i * TAU / count;
        float s = sinf(a);
        if ((s > 0) != front) {
            continue;
        }
        float rr = 25.0f + 2.0f * sinf(i * 1.9f + p->t * 0.7f);
        int x = iround(cx + cosf(a) * rr);
        int y = iround(cy - 3 + s * rr * 0.42f);
        float tw = 0.5f + 0.5f * sinf(p->t * 5.0f + i * 1.7f);
        draw_sparkle(x, y, tw, front);
    }
}

static void draw_thought_dots(float x, float y, float t)
{
    int active = (int)(fracf(t * 1.6f) * 3.0f);
    for (int i = 0; i < 3; i++) {
        int bx = iround(x + i * 4);
        int by = iround(y - i * 3) - (i == active ? 1 : 0);
        uint8_t c = i == active ? C_G0 : C_ACC;
        px(bx, by, c);
        px(bx + 1, by, c);
        px(bx, by + 1, c);
        px(bx + 1, by + 1, i == active ? C_G1 : C_G2);
    }
}

static void draw_hearts(float cx, float top, float t, float amount)
{
    static const char *const HEART[] = { ".#.#.", "#o###", "#####", ".###.", "..#.." };
    for (int i = 0; i < 2; i++) {
        float ph = fracf(t * 0.9f + i * 0.5f);
        if (ph > amount) {
            continue;
        }
        int hx = iround(cx + (i ? 13 : -18) + sinf(ph * TAU + i) * 2);
        int hy = iround(top - ph * 10);
        stamp(HEART, 5, hx, hy, C_HEART, C_WHITE);
    }
}

static void draw_alert(int x, int y)
{
    static const char *const BANG[] = { ".##.", ".##.", ".##.", ".##.", "....", ".##." };
    stamp(BANG, 6, x - 2, y, C_ACC, C_ACC);
}

/* ---------------------------------------------------------------------------
 * Frame
 * ------------------------------------------------------------------------- */

/*
 * Screen pixel -> grid cell, with 0x80 set on a cell's last screen pixel. When
 * cells are 3+ pixels, that edge is drawn dimmer so the pixel grid shows.
 */
#define MAP_MAX 512
static uint8_t s_map[MAP_MAX];
static int s_size;

void muse_pixel_set_size(int px)
{
    s_size = px < MAP_MAX ? px : MAP_MAX;
    bool grid = s_size >= 3 * W;
    for (int i = 0; i < s_size; i++) {
        int cell = i * W / s_size;
        bool edge = grid && (i + 1) * W / s_size != cell;
        s_map[i] = (uint8_t)(cell | (edge ? 0x80 : 0));
    }
}

void muse_pixel_scale(uint16_t *dst, int stride_px, int x0, int x1, int y0, int y1)
{
    int n = x1 - x0 + 1;
    const uint8_t *xmap = &s_map[x0];
    const uint16_t *prev = NULL;
    uint8_t prev_m = 0;
    for (int y = y0; y <= y1; y++, dst += stride_px) {
        uint8_t m = s_map[y];
        if (prev && m == prev_m) {
            memcpy(dst, prev, n * sizeof(uint16_t));
            continue;
        }
        const uint8_t *row = &s_fb[(m & 0x7f) * W];
        if (m & 0x80) {
            for (int i = 0; i < n; i++) {
                dst[i] = s_pal_dim[row[xmap[i] & 0x7f]];
            }
        } else {
            for (int i = 0; i < n; i++) {
                uint8_t xm = xmap[i];
                uint8_t c = row[xm & 0x7f];
                dst[i] = xm & 0x80 ? s_pal_dim[c] : s_pal[c];
            }
        }
        prev = dst;
        prev_m = m;
    }
}

void muse_pixel_render(const muse_pose_t *p)
{
    static bool s_luts;
    if (!s_luts) {
        init_luts();
        s_luts = true;
    }
    float dt = s_eyes.last_t > 0 ? clampf(p->t - s_eyes.last_t, 0, 0.2f) : 0.04f;
    s_eyes.last_t = p->t;

    muse_mode_t mode = p->mode;
    /* Powering down: the wave flattens and the eye's glow fades out. */
    float fade = mode == MUSE_MODE_OFF ? clampf(1.0f - p->mode_t / 1.3f, 0, 1) : 1.0f;
    float happy = p->happy;
    float level = clampf(p->level, 0, 1);
    float t = p->t;

    update_palette(&SCHEMES[mode], dt);
    float blink = eyes_update(p, dt);

    /* Touching the avatar fires the pet reaction; each new reaction
     * jumps the wave to a random other shape, which then stays selected.
     * (A touch during the reaction's first second only extends it -- the
     * pose carries no separate tap event -- so the jump happens when a
     * reaction starts, or restarts as the previous one eases out.) The
     * shared frand() stream also feeds the blink and gaze timers, whose
     * timing depends on the frame clock, so the picks don't repeat in a
     * fixed order. */
    if (p->happy > 0.5f && s_prev_happy <= 0.5f && mode != MUSE_MODE_ERROR) {
        int next = (int)(frand() * (WAVE_COUNT - 1));
        if (next >= s_wave_pattern) {
            next++;
        }
        s_wave_pattern = next;
    }
    s_prev_happy = p->happy;

    memset(s_fb, C_BG, sizeof(s_fb));
    memset(s_mask, 0, sizeof(s_mask));

    /* ---- body motion ---- */
    float bob, lean = 0, hop = 0;
    switch (mode) {
    case MUSE_MODE_LISTENING:
        bob = sinf(t * 3.0f) * 0.6f;
        break;
    case MUSE_MODE_THINKING:
        bob = sinf(t * 2.4f) * 0.8f;
        lean = sinf(t * 1.3f) * 1.2f;
        break;
    case MUSE_MODE_SPEAKING:
        bob = sinf(t * 5.0f) * 0.6f - level * 1.5f;
        break;
    case MUSE_MODE_ERROR:
        bob = 1.0f;
        lean = sinf(t * 18.0f) * (p->mode_t < 0.6f ? 1.0f : 0.0f);
        break;
    default:
        bob = sinf(t * 1.8f) * 1.0f;
        break;
    }
    if (happy > 0) {
        hop = fabsf(sinf(t * 9.0f)) * 3.0f * happy;
    }

    /* Boot: the disc pops up from a squash, then the eye opens. */
    float boot = mode == MUSE_MODE_BOOT ? clampf(p->mode_t / 1.4f, 0, 1) : 1.0f;
    float pop = mode == MUSE_MODE_BOOT ? clampf(p->mode_t / 0.6f, 0, 1) : 1.0f;
    float squash = 1.0f - (1.0f - pop) * 0.35f + sinf(pop * 3.1416f) * 0.06f;

    float breathe = sinf(t * 2.0f + 1.0f) * 0.03f;
    float rx = 25.0f * (1 + breathe) * (1.0f + (1.0f - squash) * 0.5f);
    float ry = 25.0f * (1 - breathe) * squash;
    float cx = 32.0f + lean;
    float cy = 32.5f + bob * 0.5f - hop;

    /* The eye wanders with the gaze; its glow pulses -- slow and calm at
     * rest, quicker with the voice, racing in ERROR. */
    float ex = cx + s_eyes.gx * 2.0f;
    float ey = cy + s_eyes.gy * 1.5f;
    float pulse_rate = mode == MUSE_MODE_SPEAKING ? 7.0f : mode == MUSE_MODE_LISTENING ? 4.0f
                     : mode == MUSE_MODE_ERROR ? 9.0f : mode == MUSE_MODE_THINKING ? 2.5f : 1.6f;
    float pulse = 0.5f + 0.5f * sinf(t * pulse_rate);
    float eye_gain = (0.80f + 0.35f * pulse + level * 0.30f) * (1.0f - blink * 0.75f) * fade;
    if (mode == MUSE_MODE_BOOT) {
        eye_gain *= p->mode_t < 0.9f ? 0.12f : clampf((p->mode_t - 0.9f) / 0.3f, 0, 1);
    }
    if (happy > 0 && mode != MUSE_MODE_ERROR) {
        eye_gain += 0.25f * happy;
    }
    float eye_r = 12.5f * (0.96f + 0.05f * pulse) * (mode == MUSE_MODE_LISTENING ? 1.06f : 1.0f);

    /* ---- background layers ---- */
    float aura_r = 29.0f + level * 4.0f + sinf(t * 1.5f) * 1.0f;
    float aura_s = (0.75f * boot + level * 0.4f) * fade;
    draw_aura(cx, cy, aura_r, aura_s);
    if (mode == MUSE_MODE_LISTENING) {
        draw_rings(cx, cy, t, level, 0.9f);
    } else if (mode == MUSE_MODE_SPEAKING) {
        draw_rings(cx, cy, t, level, 0.6f);
    }
    draw_shadow(cx, 58.0f, 12.0f - hop * 0.8f);

    float spk_speed = mode == MUSE_MODE_THINKING ? 2.8f : mode == MUSE_MODE_LISTENING ? 1.2f
                    : mode == MUSE_MODE_SPEAKING ? 1.5f : 0.6f;
    int spk_count = mode == MUSE_MODE_BOOT ? (int)(boot * 6) : (int)(6 * fade);
    draw_sparkles(p, cx, cy, false, spk_speed, spk_count);

    /* ---- the disc, its eye, and the wave over both ---- */
    draw_disc(cx, cy, rx, ry, ex, ey, eye_r, eye_gain, fade);

    float amp, phase;
    switch (mode) {
    case MUSE_MODE_LISTENING:
        amp = 3.0f + level * 7.0f;
        phase = t * 6.0f;
        break;
    case MUSE_MODE_SPEAKING:
        amp = 2.5f + level * 9.0f + 0.4f * (0.5f + 0.5f * sinf(t * 22.0f));
        phase = t * 9.0f;
        break;
    case MUSE_MODE_THINKING:
        amp = 2.2f;
        phase = t * 1.4f;
        break;
    case MUSE_MODE_ERROR:
        amp = 4.0f;
        phase = t * 3.0f;
        break;
    case MUSE_MODE_OFF:
        amp = 3.0f * fade;
        phase = t * 2.0f;
        break;
    case MUSE_MODE_BOOT:
        amp = 3.0f * boot;
        phase = t * 2.2f;
        break;
    default:
        amp = 3.2f;
        phase = t * 2.2f;
        break;
    }
    if (happy > 0 && mode != MUSE_MODE_ERROR) {
        amp += 2.0f * happy;
    }
    amp *= 1.0f - blink * 0.7f;
    draw_waveform(cx, cy, 23.0f, amp, phase,
                  mode == MUSE_MODE_ERROR ? WAVE_SQUARE : s_wave_pattern, fade,
                  mode == MUSE_MODE_ERROR ? C_TONGUE : C_WHITE);

    /* ---- foreground ---- */
    draw_sparkles(p, cx, cy, true, spk_speed, spk_count);
    if (mode == MUSE_MODE_THINKING) {
        draw_thought_dots(cx + 13, cy - 20, t);
    }
    if (happy > 0) {
        draw_hearts(cx, cy - 27, t, happy);
    }
    if (mode == MUSE_MODE_ERROR) {
        draw_alert(iround(cx + 21), iround(cy - 27));
    }
}

