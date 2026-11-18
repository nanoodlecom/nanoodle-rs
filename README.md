# nanoodle-rs

A Rust executor for [nanoodle](https://nanoodle.com) workflow graphs — the
third implementation of the same frozen graph format, alongside the
JavaScript (`nanoodle`) and Python (`nanoodle`) SDKs. It loads and runs the
exact graphs the editor and the other executors run: all 31 node types,
including chat, image, video, audio, 3D (generation and mesh processing such
as Hyper3D Rigging), Decide gates, custom endpoints, and the local media
nodes (resize, trim, frame extraction, combine, soundtrack, lipsync,
inpaint).

Built to be a core you can put anywhere: one static binary, ~3 MB, cold
process + graph load in ~2 ms, and a persistent worker-pool scheduler that
runs independent nodes concurrently.

## Build

```sh
cargo build --release
# binary: target/release/nanoodle-rs
```

## Use

```sh
# Load-only: parse + validate a graph, print the load report as JSON
nanoodle-rs load graph.json

# Run a graph against NanoGPT (or a compatible endpoint)
nanoodle-rs run graph.json --api-key <key> \
    [--base-url https://nano-gpt.com] [--catalog catalog.json]
```

Both commands print a JSON report: per-node status, outputs, cost, and
errors. A run whose sink node fails still prints the partial report and
reports the failure in the run error field.

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
