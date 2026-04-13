#include <algorithm>
#include <immintrin.h>
#include "our_gl.h"

mat<4,4> ModelView, Viewport, Perspective; // "OpenGL" state matrices
std::vector<double> zbuffer;               // depth buffer

void lookat(const vec3 eye, const vec3 center, const vec3 up) {
    vec3 n = normalized(eye-center);
    vec3 l = normalized(cross(up,n));
    vec3 m = normalized(cross(n, l));
    ModelView = mat<4,4>{{{l.x,l.y,l.z,0}, {m.x,m.y,m.z,0}, {n.x,n.y,n.z,0}, {0,0,0,1}}} *
                mat<4,4>{{{1,0,0,-center.x}, {0,1,0,-center.y}, {0,0,1,-center.z}, {0,0,0,1}}};
}

void init_perspective(const double f) {
    Perspective = {{{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0, -1/f,1}}};
}

void init_viewport(const int x, const int y, const int w, const int h) {
    Viewport = {{{w/2., 0, 0, x+w/2.}, {0, h/2., 0, y+h/2.}, {0,0,1,0}, {0,0,0,1}}};
}

void init_zbuffer(const int width, const int height) {
    zbuffer = std::vector(width*height, -1000.);
}

void rasterize(const Triangle &clip, const IShader &shader, TGAImage &framebuffer) {
    vec4 ndc[3]    = { clip[0]/clip[0].w, clip[1]/clip[1].w, clip[2]/clip[2].w };                // normalized device coordinates
    vec2 screen[3] = { (Viewport*ndc[0]).xy(), (Viewport*ndc[1]).xy(), (Viewport*ndc[2]).xy() }; // screen coordinates

    mat<3,3> ABC = {{ {screen[0].x, screen[0].y, 1.}, {screen[1].x, screen[1].y, 1.}, {screen[2].x, screen[2].y, 1.} }};
    if (ABC.det()<1) return; // backface culling + discarding triangles that cover less than a pixel
    const mat<3,3> bary = ABC.invert_transpose();
    const vec3 bc_c   = { bary[0][2], bary[1][2], bary[2][2] };
    const vec3 bc_dx  = { bary[0][0], bary[1][0], bary[2][0] };
    const vec3 bc_dy  = { bary[0][1], bary[1][1], bary[2][1] };
    const vec3 ndc_z  = { ndc[0].z, ndc[1].z, ndc[2].z };

    auto [bbminx,bbmaxx] = std::minmax({screen[0].x, screen[1].x, screen[2].x}); // bounding box for the triangle
    auto [bbminy,bbmaxy] = std::minmax({screen[0].y, screen[1].y, screen[2].y}); // defined by its top left and bottom right corners
    const int xmin = std::max<int>(bbminx, 0);
    const int xmax = std::min<int>(bbmaxx, framebuffer.width()-1);
    const int ymin = std::max<int>(bbminy, 0);
    const int ymax = std::min<int>(bbmaxy, framebuffer.height()-1);
    const int width = framebuffer.width();

    auto shade_pixel = [&](const int px, const int py, const vec3 &bc_screen, const double z) {
        if (bc_screen.x<0 || bc_screen.y<0 || bc_screen.z<0) return;
        vec3 bc_clip = { bc_screen.x/clip[0].w, bc_screen.y/clip[1].w, bc_screen.z/clip[2].w };
        bc_clip = bc_clip / (bc_clip.x + bc_clip.y + bc_clip.z);
        const int idx = px + py*width;
        if (z <= zbuffer[idx]) return;
        auto [discard, color] = shader.fragment(bc_clip);
        if (discard) return;
        zbuffer[idx] = z;
        framebuffer.set(px, py, color);
    };

#pragma omp parallel for
    for (int x=xmin; x<=xmax; x++) {         // clip the bounding box by the screen
        const double xd = static_cast<double>(x);
        vec3 bc_line = bc_c + bc_dx*xd + bc_dy*static_cast<double>(ymin);
        int y = ymin;
#if defined(__SSE2__)
        const __m128d ndc_x = _mm_set1_pd(ndc_z.x);
        const __m128d ndc_y = _mm_set1_pd(ndc_z.y);
        const __m128d ndc_zv = _mm_set1_pd(ndc_z.z);
        for (; y+1<=ymax; y+=2) {
            vec3 bc_next = bc_line + bc_dy;
            __m128d bc0 = _mm_set_pd(bc_next.x, bc_line.x);
            __m128d bc1 = _mm_set_pd(bc_next.y, bc_line.y);
            __m128d bc2 = _mm_set_pd(bc_next.z, bc_line.z);
            __m128d z_pair = _mm_mul_pd(bc0, ndc_x);
            z_pair = _mm_add_pd(z_pair, _mm_mul_pd(bc1, ndc_y));
            z_pair = _mm_add_pd(z_pair, _mm_mul_pd(bc2, ndc_zv));
            alignas(16) double z_vals[2];
            _mm_store_pd(z_vals, z_pair);
            shade_pixel(x, y, bc_line, z_vals[0]);
            shade_pixel(x, y+1, bc_next, z_vals[1]);
            bc_line = bc_next + bc_dy;
        }
#endif
        for (; y<=ymax; y++) {
            const double z = bc_line.x*ndc_z.x + bc_line.y*ndc_z.y + bc_line.z*ndc_z.z;
            shade_pixel(x, y, bc_line, z);
            bc_line = bc_line + bc_dy;
        }
    }
}
