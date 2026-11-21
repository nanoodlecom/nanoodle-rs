use nanoodle::{materialize, run, Ctx};
use serde_json::Value;

#[test]
fn splat_graph_runs_with_just_the_sheet_wire() {
    let mut data: Value = serde_json::from_str(include_str!("fixtures/sprite-stack.json")).unwrap();
    data["nodes"][2]["fields"]["turntable"] = Value::Bool(false);
    let graph = materialize(&data).unwrap();
    assert!(graph.warnings.is_empty());
    let result = run(&graph, &Ctx::new("", "")).unwrap_or_else(|e| panic!("{}", e.message));
    let sliced = &result.nodes["s"].out;
    let rendered = &result.nodes["r"].out;
    assert!(sliced["sheet"]
        .as_str()
        .unwrap()
        .starts_with("data:image/png;base64,"));
    let layout: Value = serde_json::from_str(sliced["layout"].as_str().unwrap()).unwrap();
    assert_eq!(layout["upAxis"], "z");
    assert_eq!(sliced["image"], rendered["image"]);
    assert!(!rendered.contains_key("video"));
    assert_eq!(result.cost_usd, 0.0);
}

#[test]
fn turntable_runs_with_an_empty_executable_search_path() {
    let graph = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sprite-stack.json"
    );
    let run = std::process::Command::new(env!("CARGO_BIN_EXE_nanoodle-rs"))
        .args(["run", graph])
        .env("PATH", "")
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let report: Value = serde_json::from_slice(&run.stdout).unwrap();
    assert_eq!(report["nodes"]["r"]["status"], "done", "{report}");

    let data: Value = serde_json::from_str(include_str!("fixtures/sprite-stack.json")).unwrap();
    let graph = materialize(&data).unwrap();
    let result =
        nanoodle::run(&graph, &Ctx::new("", "")).unwrap_or_else(|e| panic!("{}", e.message));
    let video = result.nodes["r"].out["video"].as_str().unwrap();
    assert!(video.starts_with("data:video/mp4;base64,"));
    assert!(video.len() > 128);
}
