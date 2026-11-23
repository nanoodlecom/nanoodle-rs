use serde_json::{json, Value};
use std::{
    fs,
    io::{self, BufRead, BufReader, Read, Write},
    net::TcpListener,
    path::PathBuf,
    process::{Command, Output},
    sync::mpsc::{self, RecvTimeoutError},
    thread,
    time::Duration,
};

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

    fn command(&self, mode: &str) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_nanoodle-rs"));
        command.arg(mode).arg(&self.0);
        command
    }

    fn invoke(&self, mode: &str) -> (i32, Value) {
        let output = self.command(mode).output().unwrap();
        let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|err| {
            panic!(
                "could not parse {mode} report: {err}; stdout: {:?}",
                output.stdout
            );
        });
        (output.status.code().unwrap(), report)
    }
}

fn assert_usage_error(output: Output, message: &str) {
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(output.stdout.is_empty(), "{:?}", output.stdout);
    assert!(stderr.contains(message), "{stderr}");
}

#[test]
fn rejects_unknown_arguments_and_missing_option_values() {
    let graph = GraphFile::new("options", "{\"nodes\":[]}");
    for (mode, args, expected) in [
        ("run", &["--unknown"][..], "unknown option --unknown"),
        ("run", &["extra"][..], "unexpected argument"),
        ("run", &["--base-url"][..], "--base-url requires a value"),
        ("run", &["--api-key"][..], "--api-key requires a value"),
        ("run", &["--api-key", ""][..], "--api-key requires a value"),
        ("run", &["--catalog"][..], "--catalog requires a value"),
        (
            "run",
            &["--catalog", "--api-key", "fixture"][..],
            "--catalog requires a value",
        ),
        (
            "load",
            &["--catalog", "catalog.json"][..],
            "load takes no options",
        ),
    ] {
        assert_usage_error(graph.command(mode).args(args).output().unwrap(), expected);
    }
}

fn with_chat_provider(check: impl FnOnce(&str)) -> Option<String> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    thread::scope(|scope| {
        let (keep_serving, stop) = mpsc::channel::<()>();
        let server = scope.spawn(move || {
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                        if stop.recv_timeout(Duration::from_millis(5)) == Err(RecvTimeoutError::Disconnected) {
                            return None;
                        }
                    }
                    Err(err) => panic!("{err}"),
                }
            };
            stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            let mut reader = BufReader::new(&mut stream);
            let mut headers = String::new();
            let mut content_length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line.trim().is_empty() {
                    break;
                }
                if let Some((name, value)) = line.split_once(':') {
                    if name.eq_ignore_ascii_case("content-length") {
                        content_length = value.trim().parse().unwrap();
                    }
                }
                headers.push_str(&line);
            }
            let mut request_body = String::new();
            reader.take(content_length).read_to_string(&mut request_body).unwrap();
            let body = r#"{"choices":[{"message":{"content":"provider output"}}]}"#;
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            Some(format!("{headers}\r\n{request_body}"))
        });
        check(&base_url);
        drop(keep_serving);
        server.join().unwrap()
    })
}

fn chat_graph(name: &str) -> GraphFile {
    GraphFile::new(
        name,
        r#"{"nodes":[{"id":"chat","type":"llm","name":"Result","fields":{"model":"fixture","prompt":"hello"}}]}"#,
    )
}

#[test]
fn rejects_bad_catalogs_before_calling_a_provider() {
    let graph = chat_graph("model-with-invalid-catalog");
    let missing = GraphFile::new("missing-catalog", "{}");
    fs::remove_file(&missing.0).unwrap();
    let invalid = GraphFile::new("invalid-catalog", "{");
    let array = GraphFile::new("array-catalog", "[]");
    let bad_section = GraphFile::new("bad-catalog-section", r#"{"chat":{}}"#);
    for (catalog, message) in [
        (missing, "cannot read --catalog"),
        (invalid, "invalid JSON in --catalog"),
        (array, "expected a JSON object"),
        (bad_section, "catalog section 'chat' must be an array"),
    ] {
        let request = with_chat_provider(|base_url| {
            let output = graph
                .command("run")
                .args(["--base-url", base_url, "--api-key", "fixture", "--catalog"])
                .arg(&catalog.0)
                .output()
                .unwrap();
            assert_usage_error(output, message);
        });
        assert!(
            request.is_none(),
            "unexpected provider request: {request:?}"
        );
    }
}

#[test]
fn run_accepts_the_documented_provider_options() {
    let graph = chat_graph("model-with-valid-catalog");
    let catalog = GraphFile::new("catalog", r#"{"chat":[{"id":"fixture"}],"metadata":{}}"#);
    let request = with_chat_provider(|base_url| {
        let output = graph
            .command("run")
            .args(["--api-key", "fixture", "--base-url", base_url, "--catalog"])
            .arg(&catalog.0)
            .output()
            .unwrap();
        assert!(output.status.success(), "{:?}", output.stderr);
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["outputs"]["Result"], "provider output");
    })
    .unwrap();
    assert!(request.starts_with("POST /api/v1/chat/completions HTTP/1.1\r\n"));
    assert!(request
        .to_ascii_lowercase()
        .contains("\r\nauthorization: bearer fixture\r\n"));
}

#[test]
fn malformed_wires_are_reported_before_a_provider_can_use_the_fallback() {
    let mut data = json!({
        "nodes": [
            {"id":"source","type":"text","fields":{"text":"wired prompt"}},
            {"id":"chat","type":"llm","fields":{
                "model":"fixture", "prompt":"stored fallback", "promtp":"also not a supported field"
            }},
        ],
        "links": [],
    });
    for (name, source, target, port, diagnostic) in [
        ("missing-source", "absent", "chat", "prompt",
            "wire absent.text → chat.prompt references missing source node \"absent\""),
        ("missing-target", "source", "absent", "prompt",
            "wire source.text → absent.prompt references missing target node \"absent\""),
        ("target-typo", "source", "chat", "promtp",
            "wire source.text → chat.promtp uses undeclared target port \"promtp\" on LLM"),
    ] {
        data["links"] = json!([{
            "from": {"node":source,"port":"text"}, "to": {"node":target,"port":port},
        }]);
        let file = GraphFile::new(name, &data.to_string());
        let (status, report) = file.invoke("load");
        assert_eq!(status, 0);
        assert_eq!(report["loadWarnings"], json!([diagnostic]));

        let request = with_chat_provider(|base_url| {
            let output = file.command("run").args(["--base-url", base_url, "--api-key", "fixture"])
                .output().unwrap();
            let report: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(output.status.code(), Some(1), "{report}");
            assert_eq!(report["runError"], diagnostic);
            assert_eq!(report["nodes"], json!({}));
            assert_eq!(report["cost"], Value::Null);
        });
        assert!(request.is_none(), "unexpected provider request: {request:?}");
    }

    data["links"] = json!([{
        "from": {"node":"source","port":"text"}, "to": {"node":"chat","port":"prompt"},
    }]);
    let file = GraphFile::new("valid-prompt-wire", &data.to_string());
    let request = with_chat_provider(|base_url| {
        let output = file.command("run").args(["--base-url", base_url, "--api-key", "fixture"])
            .output().unwrap();
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(output.status.success(), "{report}");
        assert_eq!(report["outputs"]["chat"], "provider output");
    }).expect("the valid graph should request the provider");
    let (_, body) = request.split_once("\r\n\r\n").unwrap();
    let body: Value = serde_json::from_str(body).unwrap();
    assert_eq!(body["messages"][1]["content"], "wired prompt");
}

#[test]
fn competing_prompt_wires_fail_before_the_provider_in_either_order() {
    let mut graph = json!({
        "nodes": [
            {"id":"source","type":"text","fields":{"text":"source value"}},
            {"id":"other","type":"text","fields":{"text":"other value"}},
            {"id":"request","type":"llm","fields":{"model":"fixture","prompt":"stored fallback"}},
        ],
        "links": [
            {"from":{"node":"source","port":"text"},"to":{"node":"request","port":"prompt"}},
            {"from":{"node":"other","port":"text"},"to":{"node":"request","port":"prompt"}},
        ],
    });
    let diagnostic = "input request.prompt has competing wires from other.text, source.text. Leave one active wire for this input.";
    for name in ["competing-inputs", "competing-inputs-reversed"] {
        let file = GraphFile::new(name, &graph.to_string());
        let (status, report) = file.invoke("load");
        assert_eq!(status, 0);
        assert_eq!(report["loadWarnings"], json!([diagnostic]));
        let request = with_chat_provider(|base_url| {
            let output = file.command("run").args(["--base-url", base_url, "--api-key", "fixture"])
                .output().unwrap();
            let report: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(output.status.code(), Some(1), "{report}");
            assert_eq!(report["runError"], diagnostic);
            assert_eq!(report["nodes"], json!({}));
            assert_eq!(report["cost"], Value::Null);
        });
        assert!(request.is_none(), "unexpected provider request: {request:?}");
        graph["links"].as_array_mut().unwrap().reverse();
    }

    graph["links"][1] = graph["links"][0].clone();
    let file = GraphFile::new("repeated-same-source", &graph.to_string());
    let (_, report) = file.invoke("load");
    assert_eq!(report["loadWarnings"], json!([]));
    let request = with_chat_provider(|base_url| {
        let output = file.command("run").args(["--base-url", base_url, "--api-key", "fixture"])
            .output().unwrap();
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(output.status.success(), "{report}");
        assert_eq!(report["outputs"]["request"], "provider output");
    }).expect("repeated wires from the same source should send one request");
    let (_, body) = request.split_once("\r\n\r\n").unwrap();
    let body: Value = serde_json::from_str(body).unwrap();
    assert_eq!(body["messages"][1]["content"], "source value");
}

#[test]
fn distinct_image_handles_are_collected_as_separate_inputs() {
    let file = GraphFile::new("distinct-image-inputs", &json!({
        "nodes": [
            {"id":"first","type":"upload","fields":{"image":"data:image/png;base64,YQ=="}},
            {"id":"second","type":"upload","fields":{"image":"data:image/png;base64,Yg=="}},
            {"id":"chat","type":"llm","fields":{"model":"fixture","prompt":"what is shown?"}},
        ],
        "links": [
            {"from":{"node":"first","port":"image"},"to":{"node":"chat","port":"img"}},
            {"from":{"node":"second","port":"image"},"to":{"node":"chat","port":"img1"}},
        ],
    }).to_string());
    let request = with_chat_provider(|base_url| {
        let output = file.command("run").args(["--base-url", base_url, "--api-key", "fixture"])
            .output().unwrap();
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(output.status.success(), "{report}");
        assert_eq!(report["loadWarnings"], json!([]));
    }).unwrap();
    let (_, body) = request.split_once("\r\n\r\n").unwrap();
    let body: Value = serde_json::from_str(body).unwrap();
    let mut images: Vec<_> = body["messages"][1]["content"].as_array().unwrap().iter()
        .filter_map(|part| part["image_url"]["url"].as_str()).collect();
    images.sort_unstable();
    assert_eq!(images, ["data:image/png;base64,YQ==", "data:image/png;base64,Yg=="]);
}

#[test]
fn unsupported_llm_media_wires_are_inspectable_and_stop_before_a_provider_call() {
    for (source_type, port, media) in [
        ("aupload", "audio", "data:audio/wav;base64,AAAA"),
        ("vupload", "video", "data:video/mp4;base64,AAAA"),
    ] {
        let data = json!({
            "nodes": [
                {"id":"source","type":source_type,"fields":{(port):media}},
                {"id":"chat","type":"llm","fields":{"model":"fixture","prompt":"authored prompt"}},
            ],
            "links": [{
                "from":{"node":"source","port":port}, "to":{"node":"chat","port":port},
            }],
        });
        let file = GraphFile::new(port, &data.to_string());
        let (status, report) = file.invoke("load");
        assert_eq!(status, 0);
        assert_eq!(report["loadWarnings"], json!([]));

        let request = with_chat_provider(|base_url| {
            let output = file.command("run").args(["--base-url", base_url, "--api-key", "fixture"])
                .output().unwrap();
            let report: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(output.status.code(), Some(1), "{report}");
            assert_eq!(report["runError"], format!(
                "wire source.{port} → chat.{port} uses unsupported LLM input {port:?} in the Rust executor"
            ));
            assert_eq!(report["nodes"], json!({}));
            assert_eq!(report["cost"], Value::Null);
        });
        assert!(request.is_none(), "unexpected provider request: {request:?}");
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
