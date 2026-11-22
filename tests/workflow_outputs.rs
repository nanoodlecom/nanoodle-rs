use nanoodle::{materialize, run, Ctx};
use serde_json::{json, Value};
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::TcpListener,
    sync::mpsc,
    thread,
    time::Duration,
};

fn mock_endpoint(content_type: &str, body: &str, check: impl FnOnce(String)) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    thread::scope(|scope| {
        let (_keep_accepting, stop) = mpsc::channel::<()>();
        scope.spawn(move || {
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        if stop.recv_timeout(Duration::from_millis(5)).is_err_and(|e| {
                            e == mpsc::RecvTimeoutError::Disconnected
                        }) {
                            return;
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
        });
        check(base);
    });
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
