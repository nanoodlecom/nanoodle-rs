use nanoodle::{materialize, run, topo_sort, Ctx};
use serde_json::{json, Value};
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::TcpListener,
    sync::mpsc,
    thread,
    time::Duration,
};

fn mock_endpoint(content_type: &str, body: &str, check: impl FnOnce(String)) -> bool {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    thread::scope(|scope| {
        let (keep_accepting, stop) = mpsc::channel::<()>();
        let responder = scope.spawn(move || {
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        if stop.recv_timeout(Duration::from_millis(5)).is_err_and(|e| {
                            e == mpsc::RecvTimeoutError::Disconnected
                        }) {
                            return false;
                        }
                    }
                    Err(e) => panic!("{e}"),
                }
            };
            stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            let mut reader = BufReader::new(&mut stream);
            let mut request = String::new();
            reader.read_line(&mut request).unwrap();
            assert_eq!(request.split_whitespace().nth(1), Some("/generate"));
            let mut size = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let line = line.trim();
                if line.is_empty() {
                    break;
                }
                if let Some((name, value)) = line.split_once(':') {
                    if name.eq_ignore_ascii_case("content-length") {
                        size = value.trim().parse().unwrap();
                    }
                }
            }
            io::copy(&mut reader.take(size), &mut io::sink()).unwrap();
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            true
        });
        check(base);
        drop(keep_accepting);
        responder.join().unwrap()
    })
}

#[test]
fn custom_endpoint_sinks_emit_the_effective_output_port() {
    let cases = [
        (
            Some("image"),
            None,
            "image",
            "application/json",
            r#"{"data":[{"url":"https://fixture.invalid/image.png"}]}"#,
            "https://fixture.invalid/image.png",
        ),
        (
            Some("chat"),
            Some(("mode", "video")),
            "video",
            "application/json",
            r#"{"url":"https://fixture.invalid/video.mp4"}"#,
            "https://fixture.invalid/video.mp4",
        ),
        (
            Some("image"),
            Some(("url", "audio · /generate")),
            "audio",
            "audio/mpeg",
            "fixture",
            "data:audio/mpeg;base64,Zml4dHVyZQ==",
        ),
        (
            Some("audio"),
            Some(("mode", " ")),
            "audio",
            "application/json",
            r#"{"url":"https://fixture.invalid/audio.mp3"}"#,
            "https://fixture.invalid/audio.mp3",
        ),
        (
            None,
            None,
            "text",
            "application/json",
            r#"{"choices":[{"message":{"content":"hello"}}]}"#,
            "hello",
        ),
    ];
    for (mode, wire, port, content_type, body, expected) in cases {
        mock_endpoint(content_type, body, |base| {
            let typed_path = if wire.is_some_and(|(target, _)| target == "url") {
                "/typed-path"
            } else {
                "/generate"
            };
            let mut fields = json!({"url": format!("{base}{typed_path}"), "prompt": "generate"});
            if let Some(mode) = mode {
                fields["mode"] = json!(mode);
            }
            let mut graph = json!({
                "nodes": [{"id": "sink", "type": "endpoint", "name": "Result", "fields": fields}],
                "links": [],
            });
            if let Some((target, value)) = wire {
                graph["nodes"].as_array_mut().unwrap().push(json!({
                    "id": "route", "type": "text", "fields": {"text": value, "optional": true}
                }));
                graph["links"].as_array_mut().unwrap().push(json!({
                    "from": {"node": "route", "port": "text"},
                    "to": {"node": "sink", "port": target}
                }));
            }
            let graph = materialize(&graph).unwrap();
            let result = run(&graph, &Ctx::new(&base, "offline-fixture"))
                .unwrap_or_else(|e| panic!("expected {port} output: {}", e.message));
            assert_eq!(result.nodes["sink"].out[port], expected);
            assert_eq!(result.outputs["sink"], expected);
            assert_eq!(result.outputs["Result"], expected);
        });
    }
}

#[test]
fn invalid_source_wire_warns_and_stops_before_requesting_a_provider() {
    let requested = mock_endpoint(
        "application/json",
        r#"{"choices":[{"message":{"content":"called unexpectedly"}}]}"#,
        |base| {
            let graph = materialize(&json!({
                "nodes": [
                    {"id": "source", "type": "text", "fields": {"text": "upstream"}},
                    {"id": "sink", "type": "endpoint", "fields": {
                        "url": format!("{base}/generate"), "mode": "chat", "prompt": "authored prompt"
                    }},
                ],
                "links": [{
                    "from": {"node": "source", "port": "obsoletePort"},
                    "to": {"node": "sink", "port": "mode"},
                }],
            })).unwrap();
            let diagnostic = "wire source.obsoletePort → sink.mode uses undeclared source port \"obsoletePort\" on Text";
            assert_eq!(graph.links[0].from_port, "obsoletePort");
            assert_eq!(graph.warnings, [diagnostic]);
            let failure = run(&graph, &Ctx::new(&base, "offline-fixture")).unwrap_err();
            assert_eq!(failure.message, diagnostic);
            assert!(failure.result.is_none());
        },
    );
    assert!(!requested);
}

#[test]
fn source_wire_validation_preserves_structurally_possible_and_legacy_ports() {
    for (ty, port) in [
        ("audio", "audio"),
        ("decide-branch", "yes"),
        ("decide", "decision"),
        ("vframes", "frame48"),
        ("endpoint", "audio"),
        ("endpoint", "images"),
        ("image", "images"),
        ("edit", "layers"),
        ("inpaint", "layers"),
        ("render3d", "video"),
        ("slice-3d", "layout"),
        ("sprite-stack-render", "video"),
    ] {
        let graph = materialize(&json!({
            "nodes": [
                {"id": "source", "type": ty, "fields": {"mode": "chat"}},
                {"id": "sink", "type": "text"},
            ],
            "links": [{
                "from": {"node": "source", "port": port},
                "to": {"node": "sink", "port": "text"},
            }],
        })).unwrap();
        assert!(graph.warnings.is_empty(), "{ty}.{port}: {:?}", graph.warnings);
        assert_eq!(graph.links[0].from_port, port);
    }

    let graph = materialize(&json!({
        "nodes": [
            {"id": "optional", "type": "upload", "fields": {"optional": true}},
            {"id": "sink", "type": "text"},
        ],
        "links": [{
            "from": {"node": "optional", "port": "image"},
            "to": {"node": "sink", "port": "text"},
        }],
    })).unwrap();
    let result = run(&graph, &Ctx::new("", "")).unwrap_or_else(|e| panic!("{}", e.message));
    assert_eq!(result.outputs["sink"], "");
}

#[test]
fn source_wire_validation_checks_current_canonical_frame_ports() {
    let mut graph = materialize(&json!({
        "nodes": [
            {"id": "frames", "type": "vframes"},
            {"id": "sink", "type": "resize"},
        ],
        "links": [{
            "from": {"node": "frames", "port": "frame1"},
            "to": {"node": "sink", "port": "image"},
        }],
    })).unwrap();
    assert!(graph.warnings.is_empty());
    for port in ["frame0", "frame01", "frame+1", "frame49"] {
        graph.links[0].from_port = port.into();
        let failure = run(&graph, &Ctx::new("", "")).unwrap_err();
        assert!(failure.message.contains(&format!("undeclared source port {port:?}")));
        assert!(failure.result.is_none());
    }
}

#[test]
fn retained_dangling_wire_can_be_inspected_and_repaired() {
    let mut graph = materialize(&json!({
        "nodes": [
            {"id":"source","type":"text","fields":{"text":"repaired"}},
            {"id":"sink","type":"text","fields":{"text":"fallback"}},
        ],
        "links": [{"from":{"node":"source","port":"text"}}],
    })).unwrap();
    assert_eq!(graph.links.len(), 1);
    assert_eq!(graph.links[0].to_node, "");
    assert!(graph.warnings[0].contains("references missing target node"));
    assert_eq!(topo_sort(&graph).unwrap(), ["source", "sink"]);
    assert!(run(&graph, &Ctx::new("", "")).unwrap_err().result.is_none());

    graph.links[0].to_node = "sink".into();
    graph.links[0].to_port = "text".into();
    let result = run(&graph, &Ctx::new("", "")).unwrap_or_else(|e| panic!("{}", e.message));
    assert_eq!(result.outputs["sink"], "repaired");
}

#[test]
fn target_wire_validation_uses_the_declared_controls_and_handles() {
    for (ty, port, valid, effective_port) in [
        ("audio", "text", true, "prompt"),
        ("music", "text", true, "prompt"),
        ("resize", "width", true, "width"),
        ("llm", "audio", true, "audio"),
        ("llm", "video", true, "video"),
        ("llm", "img3", true, "img3"),
        ("llm", "system", true, "system"),
        ("llm", "model", true, "model"),
        ("llm", "text", false, "text"),
        ("llm", "image3", false, "image3"),
        ("edit", "image3", true, "image3"),
        ("decide-branch", "text", true, "text"),
        ("decide-branch", "branch", false, "branch"),
        ("combine", "clip2", true, "clip2"),
        ("combine", "vid3", true, "vid3"),
        ("tvideo", "ref3", true, "ref3"),
        ("ivideo", "endframe", true, "endframe"),
        ("vedit", "image", true, "image"),
        ("vedit", "ref2", true, "ref2"),
        ("render3d", "mode", true, "mode"),
        ("vframes", "stride", true, "stride"),
        ("endpoint", "mode", true, "mode"),
        ("endpoint", "url", true, "url"),
        ("comment", "audio", false, "audio"),
    ] {
        let graph = materialize(&json!({
            "nodes": [
                {"id":"source","type":"text","fields":{"text":"wired"}},
                {"id":"sink","type":ty,"fields":{"mode":"chat"}},
            ],
            "links": [{
                "from":{"node":"source","port":"text"}, "to":{"node":"sink","port":port},
            }],
        })).unwrap();
        assert_eq!(graph.links[0].to_port, effective_port);
        assert_eq!(graph.warnings.is_empty(), valid, "{ty}.{port}: {:?}", graph.warnings);
    }
}

#[test]
fn legacy_input_aliases_share_one_destination() {
    let graph = materialize(&json!({
        "nodes": [
            {"id":"first","type":"text","fields":{"text":"first prompt"}},
            {"id":"second","type":"text","fields":{"text":"second prompt"}},
            {"id":"speech","type":"audio","fields":{"model":"fixture"}},
        ],
        "links": [
            {"from":{"node":"first","port":"text"},"to":{"node":"speech","port":"text"}},
            {"from":{"node":"second","port":"text"},"to":{"node":"speech","port":"prompt"}},
        ],
    })).unwrap();
    let diagnostic = "input speech.prompt has competing wires from first.text, second.text. Leave one active wire for this input.";
    assert_eq!(graph.warnings, [diagnostic]);
    let failure = run(&graph, &Ctx::new("", "")).unwrap_err();
    assert_eq!(failure.message, diagnostic);
    assert!(failure.result.is_none());
}

#[test]
fn a_missing_or_error_skipped_output_does_not_replace_a_live_wire() {
    assert!(mock_endpoint(
        "application/json",
        r#"{"choices":[{"message":{"content":"text output"}}]}"#,
        |base| {
            let graph = materialize(&json!({
                "nodes": [
                    {"id":"source","type":"text","fields":{"text":"live input"}},
                    {"id":"skipped","type":"resize","fields":{"onError":"skip"}},
                    {"id":"endpoint","type":"endpoint","fields":{
                        "url":format!("{base}/generate"),"mode":"chat","prompt":"generate text"
                    }},
                    {"id":"sink","type":"text"},
                ],
                "links": [
                    {"from":{"node":"source","port":"text"},"to":{"node":"sink","port":"text"}},
                    {"from":{"node":"endpoint","port":"image"},"to":{"node":"sink","port":"text"}},
                    {"from":{"node":"skipped","port":"image"},"to":{"node":"sink","port":"text"}},
                ],
            })).unwrap();
            assert!(graph.warnings.is_empty());
            let result = run(&graph, &Ctx::new("", "")).unwrap_or_else(|e| panic!("{}", e.message));
            assert_eq!(result.outputs["sink"], "live input");
            assert_eq!(result.nodes["skipped"].status, "skipped");
            assert!(result.nodes["skipped"].error.is_some());
        },
    ));
}

fn skipped_image(id: &str) -> Value {
    json!({"id": id, "type": "image", "name": "Output", "fields": {
        "prompt": "draw a box", "onError": "skip"
    }})
}

#[test]
fn all_error_skipped_sinks_fail_with_partial_result() {
    let graph = materialize(&json!({
        "nodes": [skipped_image("first"), skipped_image("second"), {"id": "note", "type": "comment"}],
        "links": []
    })).unwrap();
    let failure = run(&graph, &Ctx::new("http://127.0.0.1:1", "offline-fixture")).unwrap_err();
    assert!(failure
        .message
        .contains("every output was skipped after failing"));
    assert!(failure.message.contains("\"Output\": pick a model first"));
    assert!(failure.message.contains("\"Output 2\": pick a model first"));
    let partial = failure.result.unwrap();
    assert!(partial.outputs.is_empty());
    assert_eq!(partial.errors.len(), 2);
    assert_eq!(partial.nodes["first"].status, "skipped");
    assert_eq!(partial.nodes["second"].status, "skipped");
}

#[test]
fn an_error_skip_keeps_other_outputs_and_their_keys() {
    let graph = materialize(&json!({
        "nodes": [
            skipped_image("first"),
            {"id": "second", "type": "text", "name": "Output", "fields": {"text": "kept"}}
        ],
        "links": []
    }))
    .unwrap();
    let result = run(&graph, &Ctx::new("http://127.0.0.1:1", "offline-fixture"))
        .unwrap_or_else(|e| panic!("{}", e.message));
    assert_eq!(result.errors.len(), 1);
    assert!(result.outputs.get("Output").is_none());
    assert_eq!(result.outputs["Output 2"], "kept");
    assert_eq!(result.outputs["second"], "kept");
}
