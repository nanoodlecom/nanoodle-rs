#ifndef NANOODLE_AVIF_H
#define NANOODLE_AVIF_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Decode the primary/first image to straight-alpha 8-bit RGBA. Each invocation
 * owns its decoder. On success release *rgba with nanoodle_avif_free. On
 * failure dimensions are zero and *rgba is null. Input remains caller-owned. */
int nanoodle_avif_decode(const unsigned char *data, size_t len, int *w, int *h, unsigned char **rgba);
const char *nanoodle_avif_error(int code);
void nanoodle_avif_free(unsigned char *rgba);
#ifdef __cplusplus
}
#endif
#endif
