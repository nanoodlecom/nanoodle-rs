# Nanoodle AVIF decoder sources

This decoder-only source pack combines dav1d's portable C AV1 decoder with
libavif's HEIF parsing, grid composition, alpha plane decoding and RGB
conversion. Nanoodle's wrapper exports a first-frame RGBA interface in
`nanoodle_avif.h`. Each decode creates its own decoder and enforces a
16,777,216-pixel limit and the sprite image dimensions before allocating output.

The following pinned GitHub source archives supply the unchanged upstream
files in this directory. SHA-256 identifies the downloaded archive.

| Component | Commit | Archive SHA-256 |
|---|---|---|
| [libavif 1.4.2](https://github.com/AOMediaCodec/libavif/tree/c5240fc79fe5c2407e10afd35f5505ef6333ea49) | c5240fc79fe5c2407e10afd35f5505ef6333ea49 | 5e9aa73e43af8021a57d744712ee0bc145267863afb7c9b118723a15dbd4cfbf |
| [dav1d 1.5.5](https://github.com/videolan/dav1d/tree/6c7bee685677fd997a431302be4fea8ea4017606) | 6c7bee685677fd997a431302be4fea8ea4017606 | 725d0ebe5bd19316aaaeee84a170b0f6f3930fc7cf53fa0c5bbe3c86ba1c2b15 |

The imported scaling subset from libyuv is part of the pinned libavif source.
See `libavif/LICENSE`, `libavif/third_party/README.md`,
`libavif/third_party/libyuv/AUTHORS`, `dav1d/COPYING` and `dav1d/doc/PATENTS`.
Preserve the notices in source and binary distributions.

## Compilation

Each nonempty line of `sources.txt` names one C translation unit relative to
this directory. Each line of `includes.txt` specifies a corresponding include
directory. The units in `units/` establish target-independent source
configuration. They compile both dav1d bit depths and select the dav1d
backend in libavif. Scalar C omits runtime CPU dispatch and assembly.
Config flags disable logging and platform-specific affinity APIs. The public
API returns an allocated block whose ownership transfers to the caller.

Use C11 (GNU/MSVC extensions as supported by the respective upstreams),
optimization, and NDEBUG. Use the platform C math and threading libraries
(libm and pthread on Unix). On macOS enable _DARWIN_C_SOURCE and on Linux
_GNU_SOURCE. On MSVC prepend `dav1d/include/compat/msvc` to the include
directories, pass /std:c11, and include the Windows thread unit already in the
manifest. Upstream dav1d provides both the SRWLOCK-based Win32 adapter and
the MSVC atomics shim. Only native Linux x86_64 compilation and execution have
been verified with this pack.

The library decodes the first primary picture/animation frame. The return buffer
contains raw picture samples. Presentation cropping, rotation and mirroring
metadata are available in libavif should a caller require those transforms.
