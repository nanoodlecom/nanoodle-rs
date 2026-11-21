/* The only Nanoodle entry points. libavif handles ISOBMFF metadata, colour and
 * alpha; its dav1d backend applies the same frame pixel limit to AV1 frames. */
#include "nanoodle_avif.h"
#include "avif/avif.h"
#include <stdlib.h>

static int dimensions_ok(const avifImage *image) {
    return image && image->width && image->height && image->height <= 8192 &&
        (image->width <= 8192 || (image->width <= 16384 && image->height <= 128)) &&
        image->width <= 16777216U / image->height;
}

const char *nanoodle_avif_error(int code) {
    switch (code) {
    case 0: return "OK";
    case 1: return "Invalid AVIF input";
    case 2: return "AVIF exceeds supported image dimensions";
    case 3: return "Could not allocate AVIF pixels";
    case 4: return "Could not decode AVIF image";
    case 5: return "Could not convert AVIF pixels to RGBA";
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
    d->codecChoice = AVIF_CODEC_CHOICE_DAV1D;
    d->maxThreads = 1;
    d->ignoreExif = AVIF_TRUE;
    d->ignoreXMP = AVIF_TRUE;
    d->imageSizeLimit = 16777216;
    d->imageDimensionLimit = 16384;
    d->imageCountLimit = 4096;
    int result = 4;
    unsigned char *pixels = NULL;
    if (avifDecoderSetIOMemory(d, data, len) != AVIF_RESULT_OK ||
        avifDecoderParse(d) != AVIF_RESULT_OK) goto done;
    if (!dimensions_ok(d->image)) { result = 2; goto done; }
    if (avifDecoderNextImage(d) != AVIF_RESULT_OK) goto done;
    if (!dimensions_ok(d->image)) { result = 2; goto done; }
    avifRGBImage rgb;
    avifRGBImageSetDefaults(&rgb, d->image);
    rgb.format = AVIF_RGB_FORMAT_RGBA;
    rgb.depth = 8;
    rgb.alphaPremultiplied = AVIF_FALSE;
    rgb.rowBytes = rgb.width * 4;
    pixels = malloc((size_t)rgb.rowBytes * rgb.height);
    if (!pixels) { result = 3; goto done; }
    rgb.pixels = pixels;
    if (avifImageYUVToRGB(d->image, &rgb) != AVIF_RESULT_OK) { result = 5; goto done; }
    *rgba = pixels;
    *w = (int)rgb.width;
    *h = (int)rgb.height;
    pixels = NULL;
    result = 0;
done:
    free(pixels);
    avifDecoderDestroy(d);
    return result;
}
