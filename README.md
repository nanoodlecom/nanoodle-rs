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

Once the first release is on crates.io, install the `nanoodle-rs` command.

```sh
cargo install nanoodle --locked
```

You can also install the CLI directly from GitHub.

```sh
cargo install --git https://github.com/nanoodlecom/nanoodle-rs.git --locked
```

To use the library in a Rust project, add the crate once it is on crates.io.

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
Use `--base-url` for a compatible provider or `--catalog catalog.json` to
supply a local model catalog.

## Sprite stacking

Wire `mupload.model` to `slice-3d.model`, then `slice-3d.sheet` to `sprite-stack-render.sheet`. A static triangle GLB, self-contained glTF or uncompressed Gaussian PLY produces a transparent PNG strip, JSON layout text and a preview. The renderer takes the strip alone. It emits a PNG and an optional MP4 turntable. The Rust executor reads PNG, JPEG, GIF, WebP, BMP, ICO, CUR and AVIF textures and palettes. glTF materials can select WebP images with `EXT_texture_webp`. Rust handles slicing, projection and H.264 turntable encoding. The crate compiles its AVIF decoder into the executor. These two nodes need only the installed executor.

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

- A node whose upstream errored is itself recorded `error`
  (`upstream failed: <name>`), cascading down the chain; a closed Decide
  gate instead skips its downstream, unbilled.
- `fields.onError: "skip"` records a failure as `skipped` and lets
  dependents run on their remaining inputs.
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
