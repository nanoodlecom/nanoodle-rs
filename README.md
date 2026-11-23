# nanoodle

A Rust executor for [nanoodle](https://nanoodle.com) workflow graphs — the
third implementation of the same frozen graph format, alongside the
JavaScript (`nanoodle`) and Python (`nanoodle`) SDKs. It runs saved editor graphs for
chat, image, video, audio, 3D (generation and mesh processing such
as Hyper3D Rigging), Decide gates, custom endpoints, and the local media
nodes (resize, trim, frame extraction, combine, soundtrack, lipsync,
inpaint, Slice 3D and Sprite stack render).

The compiled CLI embeds the sprite renderer and runs independent nodes concurrently.

## Install

Install the `nanoodle-rs` command from crates.io.

```sh
cargo install nanoodle --locked
```

You can also install the CLI directly from GitHub.

```sh
cargo install --git https://github.com/nanoodlecom/nanoodle-rs.git --locked
```

To use the library in a Rust project, add the crate.

```sh
cargo add nanoodle
```

## Run your first workflow

1. Export a workflow from [Nanoodle](https://nanoodle.com) as a JSON file and
   save it as `workflow.json`.
2. Validate the file locally. The JSON report includes the node count and any
   load warnings.

   ```sh
   nanoodle-rs load workflow.json
   ```

3. Run the workflow. For graphs that call hosted models, enter your
   [NanoGPT](https://nano-gpt.com) API key when prompted.

   ```bash
   read -rsp 'NanoGPT API key: ' NANOGPT_API_KEY
   printf '\n'
   nanoodle-rs run workflow.json --api-key "$NANOGPT_API_KEY"
   ```

The run report includes the outputs, node statuses, cost, and errors.
If the run fails, it includes the error and any results produced so far.
Load and run failures exit with status 1 and still write a JSON report to stdout.
Use `--base-url` for a compatible provider or `--catalog catalog.json` to
supply a local model catalog. Catalog files must contain a JSON object with
arrays for any model sections they include, such as `chat` or `video`.
Invalid arguments or catalog files exit with status 2 and write a diagnostic to stderr.

## Sprite stacking

Connect the 3D model or upload node’s `model` output to `slice-3d.model`, then connect `slice-3d.sheet` to `sprite-stack-render.sheet`. Slice 3D accepts static and rigged triangle GLB, self-contained glTF and uncompressed Gaussian PLY. With the default fields, it slices the rest pose and produces a transparent PNG strip, a preview and JSON layout.

For a moving character, set `preset` to `character`, `clip` to the exact glTF animation name (for example, `Walk`), and `frames` to the number of poses (2–64). A blank clip selects the first animation when requesting multiple poses. An unnamed animation uses its one-based index, such as `Animation 1`. A named clip with `frames: 1` gives its starting pose. Every pose uses the same world bounds and ground height. The character preset supplies 128-pixel tiles, 80 layers, source colors and an outline. Individual fields override the preset.

The animated sheet is an APNG that carries one complete horizontal strip per pose. The layout includes the clip name, duration and sampled times in seconds. Sprite stack render plays the sheet as an H.264 MP4 at the selected yaw and clip timing. Enable `turntable` to rotate the camera while the poses loop over the configured video duration. Its `image` port returns the first view in either mode.

The Rust executor reads PNG, JPEG, GIF, WebP, BMP, ICO, CUR and AVIF textures and palettes. glTF materials can select WebP images with `EXT_texture_webp`. The crate includes the codecs for slicing, projection and video generation. These two nodes need only the installed executor.

## Build from source

```sh
git clone https://github.com/nanoodlecom/nanoodle-rs.git
cd nanoodle-rs
cargo build --release --locked
```

The binary is at `target/release/nanoodle-rs`.

## Compatibility

The graph format is frozen — this executor changes nothing about it. Where
the JavaScript and Python executors diverge, nanoodle-rs follows the
JavaScript SDK, the reference implementation. Compatibility is proven, not
asserted: a shared three-engine conformance suite (125 cases — hand-built
node cases, the full awesome-noodles and editor-gallery corpora, catalog
variants, and failure/gating semantics) runs the same graphs through all
three executors against a scripted mock and diffs requests, outputs, node
statuses, and costs. nanoodle-rs is green across the whole suite, and two
fuzzers (synthetic graphs and mutations of real graphs) run the same
three-way comparison.

Execution semantics worth knowing:

- A malformed node record stops loading. The error identifies its position in
  the nodes array and the required shape of the record, ID, or type.
- Loading a workflow reports wires with missing nodes or undeclared source or
  destination ports. The run stops before any node executes if a wire is invalid.
- An input accepts one active source. Rust reports competing wires before the
  consuming step runs and allows mutually exclusive Decide routes to share an input.
- LLM audio and video wires are recognized on load and stop Rust runs before any
  node executes. The editor and player can use those inputs.
- A node whose upstream errored is itself recorded `error`
  (`upstream failed: <name>`), cascading down the chain; a closed Decide
  gate instead skips its downstream, unbilled.
- `fields.onError: "skip"` records a failure as `skipped` and lets
  dependents run on their remaining inputs.
- Video → frames uses `ffmpeg` and `ffprobe` on `PATH` and exposes up to 48
  image ports (`frame1` through `frame48`). A saved wire to a higher frame
  raises the requested count up to that limit. Set `startSec` or `endSec`
  for an inclusive range measured from the first video frame. Within the
  range, `stride` selects every Nth frame in the chosen direction. If you
  leave stride blank, `gap` is the minimum time between selected frames.
  The node can return fewer frames than requested when the range runs out.
- Every model-driven runner requires `fields.model` — there are no silent
  defaults (exceptions: model3d defaults to Tripo, cleanvoice has a
  default model).

### Conditional Decide

Use the `decide-branch` type to select a Yes or No route. The active port forwards the
input text or, without text, the answer. Inactive routes skip without provider charges,
joins receive active inputs, and an invalid decision opens neither route. Existing
`decide` gates keep their previous behavior. The shared graph in
`tests/fixtures/decider-branches.json` runs in the offline integration test.

## License

MIT — see [LICENSE](LICENSE).

The packaged AVIF decoder includes libavif, dav1d and libyuv source with their upstream licenses. See [codec notices and revisions](vendor/avif/README.md).
