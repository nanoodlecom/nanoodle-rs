/* The only Nanoodle entry points. libavif handles ISOBMFF metadata, colour and
 * alpha; its dav1d backend applies the same frame pixel limit to AV1 frames. */
#include "nanoodle_avif.h"
#include "avif/avif.h"
#include <stdlib.h>
#include <string.h>

static int dimensions_ok(uint32_t width, uint32_t height) {
    return width && height && height <= 8192 &&
        (width <= 8192 || (width <= 16384 && height <= 128)) &&
        width <= 16777216U / height;
}

/* HEIF quarter turns are counter-clockwise. The optional mirror is applied
 * to the rotated image, with its axis expressed in displayed coordinates. */
static int present_rgba(const avifImage *image, const avifRGBImage *rgb, unsigned char **pixels, int *w, int *h) {
    const unsigned angle = (image->transformFlags & AVIF_TRANSFORM_IROT) ? image->irot.angle : 0;
    const int mirror = (image->transformFlags & AVIF_TRANSFORM_IMIR) ? (int)image->imir.axis : -1;
    if (angle > 3 || mirror > 1) return 6;
    const uint32_t dw = (angle & 1) ? rgb->height : rgb->width;
    const uint32_t dh = (angle & 1) ? rgb->width : rgb->height;
    if (!dimensions_ok(dw, dh)) return 2;
    if (angle || mirror >= 0) {
        unsigned char *out = malloc((size_t)dw * dh * 4);
        if (!out) return 3;
        for (uint32_t y = 0; y < rgb->height; ++y) {
            const unsigned char *row = *pixels + (size_t)y * rgb->rowBytes;
            for (uint32_t x = 0; x < rgb->width; ++x) {
                uint32_t dx, dy;
                switch (angle) {
                case 1: dx = y; dy = dh - 1 - x; break;
                case 2: dx = dw - 1 - x; dy = dh - 1 - y; break;
                case 3: dx = dw - 1 - y; dy = x; break;
                default: dx = x; dy = y; break;
                }
                if (mirror == 0) dy = dh - 1 - dy;
                if (mirror == 1) dx = dw - 1 - dx;
                memcpy(out + ((size_t)dy * dw + dx) * 4, row + (size_t)x * 4, 4);
            }
        }
        free(*pixels);
        *pixels = out;
    }
    *w = (int)dw;
    *h = (int)dh;
    return 0;
}

const char *nanoodle_avif_error(int code) {
    switch (code) {
    case 0: return "OK";
    case 1: return "Invalid AVIF input";
    case 2: return "AVIF exceeds supported image dimensions";
    case 3: return "Could not allocate AVIF pixels";
    case 4: return "Could not decode AVIF image";
    case 5: return "Could not convert AVIF pixels to RGBA";
    case 6: return "Invalid AVIF presentation properties";
    default: return "Unknown AVIF error";
    }
}

void nanoodle_avif_free(unsigned char *rgba) { free(rgba); }

int nanoodle_avif_decode(const unsigned char *data, size_t len, int *w, int *h, unsigned char **rgba) {
    if (!w || !h || !rgba) return 1;
    *w = *h = 0;
    *rgba = NULL;
    if (!data || !len) return 1;
    avifDecoder *d = avifDecoderCreate();
    if (!d) return 3;
    // Chromium falls back to the full image for unrepresentable apertures.
    d->strictFlags &= ~AVIF_STRICT_CLAP_VALID;
    d->codecChoice = AVIF_CODEC_CHOICE_DAV1D;
    d->maxThreads = 1;
    d->ignoreExif = AVIF_TRUE;
    d->ignoreXMP = AVIF_TRUE;
    d->imageSizeLimit = 16777216;
    d->imageDimensionLimit = 16384;
    d->imageCountLimit = 4096;
    int result = 4;
    unsigned char *pixels = NULL;
    avifImage *view = NULL;
    if (avifDecoderSetIOMemory(d, data, len) != AVIF_RESULT_OK ||
        avifDecoderParse(d) != AVIF_RESULT_OK) goto done;
    if (!d->image || !dimensions_ok(d->image->width, d->image->height)) { result = 2; goto done; }
    if (avifDecoderNextImage(d) != AVIF_RESULT_OK) goto done;
    if (!d->image || !dimensions_ok(d->image->width, d->image->height)) { result = 2; goto done; }
    const avifImage *source = d->image;
    if (source->transformFlags & AVIF_TRANSFORM_CLAP) {
        avifCropRect crop;
        avifDiagnostics diagnostics;
        // Chrome only displays valid clean apertures anchored at the top left.
        // Set a YUV view before conversion so subsampled edges match the crop.
        if (avifCropRectFromCleanApertureBox(&crop, &source->clap, source->width, source->height, &diagnostics) &&
            crop.x == 0 && crop.y == 0 && (crop.width != source->width || crop.height != source->height)) {
            view = avifImageCreateEmpty();
            if (!view) { result = 3; goto done; }
            if (avifImageSetViewRect(view, source, &crop) != AVIF_RESULT_OK) { result = 6; goto done; }
            source = view;
        }
    }
    avifRGBImage rgb;
    avifRGBImageSetDefaults(&rgb, source);
    rgb.format = AVIF_RGB_FORMAT_RGBA;
    rgb.depth = 8;
    rgb.alphaPremultiplied = AVIF_FALSE;
    rgb.rowBytes = rgb.width * 4;
    pixels = malloc((size_t)rgb.rowBytes * rgb.height);
    if (!pixels) { result = 3; goto done; }
    rgb.pixels = pixels;
    if (avifImageYUVToRGB(source, &rgb) != AVIF_RESULT_OK) { result = 5; goto done; }
    result = present_rgba(d->image, &rgb, &pixels, w, h);
    if (result != 0) goto done;
    *rgba = pixels;
    pixels = NULL;
done:
    free(pixels);
    if (view) avifImageDestroy(view);
    avifDecoderDestroy(d);
    return result;
}
