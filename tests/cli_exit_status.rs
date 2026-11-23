use serde_json::{json, Value};
use std::{fs, path::PathBuf, process::Command};

struct GraphFile(PathBuf);

impl GraphFile {
    fn new(name: &str, contents: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "nanoodle-cli-test-{}-{name}.json",
            std::process::id()
        ));
        fs::write(&path, contents).unwrap();
        Self(path)
    }

    fn invoke(&self, mode: &str) -> (i32, Value) {
        let output = Command::new(env!("CARGO_BIN_EXE_nanoodle-rs"))
            .arg(mode)
            .arg(&self.0)
            .output()
            .unwrap();
        let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|err| {
            panic!(
                "could not parse {mode} report: {err}; stdout: {:?}",
                output.stdout
            );
        });
        (output.status.code().unwrap(), report)
    }
}

impl Drop for GraphFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[test]
fn invalid_file_exits_with_failure_and_reports_the_load_error() {
    for (name, contents, expected_message) in [
        ("invalid-json", "{", "invalid JSON:"),
        ("invalid-graph", "[]", "not a nanoodle graph:"),
    ] {
        let file = GraphFile::new(name, contents);
        for mode in ["load", "run"] {
            let (status, report) = file.invoke(mode);
            let error = report["loadError"].as_str().unwrap();
            assert!(error.starts_with(expected_message), "{report}");
            assert_eq!(report, json!({"loadError": error}));
            assert_eq!(status, 1, "{mode} {name}: {report}");
        }
    }
}

#[test]
fn failed_run_exits_with_failure_and_keeps_partial_results() {
    let file = GraphFile::new(
        "failed-resize",
        r#"{
            "nodes": [
                {"id": "resize", "type": "resize", "name": "Resize"},
                {"id": "message", "type": "text", "name": "Result", "fields": {"text": "hello"}}
            ],
            "links": []
        }"#,
    );
    let (status, report) = file.invoke("run");
    assert_eq!(
        report,
        json!({
            "loadWarnings": [],
            "runError": "run failed — \"Resize\": no image input",
            "outputs": {"message": "hello", "Result": "hello"},
            "nodes": {
                "resize": {"status": "error", "error": "no image input"},
                "message": {"status": "done", "error": null},
            },
            "cost": 0.0,
            "errors": ["node resize: no image input"],
        })
    );
    assert_eq!(status, 1);
}

#[test]
fn successful_load_and_run_exit_with_success() {
    let file = GraphFile::new(
        "text",
        r#"{
            "nodes": [{"id": "message", "type": "text", "name": "Result", "fields": {"text": "hello"}}],
            "links": []
        }"#,
    );
    let (status, report) = file.invoke("load");
    assert_eq!(status, 0);
    assert_eq!(
        report,
        json!({"loadWarnings": [], "loadOnly": {"nodeCount": 1, "types": ["text"], "nWarnings": 0}})
    );

    let (status, report) = file.invoke("run");
    assert_eq!(status, 0);
    assert_eq!(
        report,
        json!({
            "loadWarnings": [],
            "outputs": {"message": "hello", "Result": "hello"},
            "nodes": {"message": {"status": "done", "error": null}},
            "cost": 0.0,
            "errors": [],
        })
    );
}
