use super::{materialize, poll_loop, run, run_node, Ctx, PollStep, RunResult};
use serde_json::{json, Value};
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::TcpListener,
    sync::mpsc::{self, RecvTimeoutError},
    thread,
    time::{Duration, Instant},
};

struct Reply {
    status: u16,
    body: &'static str,
    headers: &'static str,
    length: Option<usize>,
    stall: bool,
}

impl Reply {
    fn json(status: u16, body: &'static str) -> Self {
        Self {
            status,
            body,
            headers: "",
            length: None,
            stall: false,
        }
    }
}

fn with_provider(replies: Vec<Reply>, check: impl FnOnce(&Ctx)) -> Vec<String> {
    with_provider_requests(replies, check).into_iter().map(|(route, _)| route).collect()
}

fn with_provider_requests(replies: Vec<Reply>, check: impl FnOnce(&Ctx)) -> Vec<(String, Vec<u8>)> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let ctx = Ctx::new(
        &format!("http://{}", listener.local_addr().unwrap()),
        "fixture",
    );
    thread::scope(|scope| {
        let (keep_serving, stop) = mpsc::channel::<()>();
        let server = scope.spawn(move || {
            let mut requests = Vec::new();
            for reply in replies {
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                            if stop.recv_timeout(Duration::from_millis(3)) == Err(RecvTimeoutError::Disconnected) {
                                return requests;
                            }
                        }
                        Err(e) => panic!("{e}"),
                    }
                };
                stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
                let mut reader = BufReader::new(&mut stream);
                let mut request = String::new();
                reader.read_line(&mut request).unwrap();
                let route = request.split_whitespace().take(2).collect::<Vec<_>>().join(" ");
                let mut length = 0;
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).unwrap();
                    if header.trim().is_empty() {
                        break;
                    }
                    if let Some((name, value)) = header.split_once(':') {
                        if name.eq_ignore_ascii_case("content-length") {
                            length = value.trim().parse().unwrap();
                        }
                    }
                }
                let mut body = Vec::new();
                reader.take(length).read_to_end(&mut body).unwrap();
                requests.push((route, body));
                write!(stream, "HTTP/1.1 {} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n{}",
                    reply.status, reply.length.unwrap_or(reply.body.len()), reply.headers, reply.body).unwrap();
                if reply.stall {
                    let _ = stop.recv_timeout(Duration::from_secs(2));
                }
            }
            requests
        });
        check(&ctx);
        drop(keep_serving);
        server.join().unwrap()
    })
}

fn node_run(node_type: &str, ctx: &Ctx) -> std::result::Result<RunResult, super::RunFailure> {
    let graph = materialize(&json!({
        "nodes": [{"id": "work", "type": node_type, "fields": {"model": "test", "prompt": "hello"}}],
        "links": []
    })).unwrap();
    run(&graph, ctx)
}

#[test]
fn custom_endpoint_prompt_can_be_supplied_by_a_text_wire() {
    let reply = r#"{"choices":[{"message":{"content":"offline reply"}}]}"#;
    for port in ["text", "prompt"] {
        let requests = with_provider_requests(vec![Reply::json(200, reply)], |ctx| {
            let graph = materialize(&json!({
                "nodes":[
                    {"id":"source","type":"text","fields":{"text":"wired request"}},
                    {"id":"sink","type":"endpoint","fields":{"url":format!("{}/generate", ctx.base_url),"mode":"chat"}},
                ],
                "links":[{"from":{"node":"source","port":"text"},"to":{"node":"sink","port":port}}],
            })).unwrap();
            let result = run(&graph, ctx).unwrap_or_else(|error| panic!("{}", error.message));
            assert_eq!(result.outputs["sink"], "offline reply");
        });
        let [(route, body)] = requests.as_slice() else {
            panic!("expected one request, got {}", requests.len());
        };
        assert_eq!(route, "POST /generate");
        let payload: Value = serde_json::from_slice(body).unwrap();
        assert_eq!(payload["messages"][0], json!({"role":"user","content":"wired request"}));
    }

    let requests = with_provider(vec![Reply::json(200, reply)], |ctx| {
        let graph = materialize(&json!({
            "nodes":[{"id":"sink","type":"endpoint","fields":{"url":format!("{}/generate", ctx.base_url)}}],
        })).unwrap();
        let failure = run(&graph, ctx).unwrap_err();
        assert!(failure.message.contains("missing required input"), "{}", failure.message);
    });
    assert!(requests.is_empty(), "{requests:?}");
}

#[test]
fn provider_errors_retain_known_cost_without_billing_rejected_requests() {
    let pricing = "x-nanogpt-pricing: {\"costUsd\":0.21}\r\n";
    let cases = [
        (
            "llm",
            "invalid JSON response",
            Some(0.21),
            Reply {
                headers: pricing,
                ..Reply::json(200, "{")
            },
        ),
        (
            "llm",
            "response read failed",
            Some(0.21),
            Reply {
                headers: pricing,
                length: Some(80),
                ..Reply::json(200, "{")
            },
        ),
        (
            "llm",
            "no text in response",
            Some(0.25),
            Reply {
                headers: pricing,
                ..Reply::json(200, r#"{"choices":[],"cost":0.25}"#)
            },
        ),
        (
            "image",
            "no image in response",
            Some(0.27),
            Reply {
                headers: pricing,
                ..Reply::json(200, r#"{"data":[],"cost":0.27}"#)
            },
        ),
        (
            "tts",
            "invalid JSON response",
            Some(0.22),
            Reply {
                headers: "x-cost: 0.22\r\n",
                ..Reply::json(200, "{")
            },
        ),
        (
            "llm",
            "API key rejected",
            None,
            Reply {
                headers: pricing,
                ..Reply::json(401, "unauthorized")
            },
        ),
    ];
    for (node_type, expected_error, cost, reply) in cases {
        let requests = with_provider(vec![reply], |ctx| {
            let error = node_run(node_type, ctx).unwrap_err();
            assert!(error.message.contains(expected_error), "{}", error.message);
            let result = error.result.unwrap();
            assert_eq!(result.nodes["work"].cost, cost);
            assert_eq!(result.cost_usd, cost.unwrap_or_default());
        });
        assert_eq!(requests.len(), 1);
    }
}

#[test]
fn audio_auth_refusal_stops_polling_and_keeps_the_submit_charge() {
    for status in [401, 402, 403] {
        let mut submit = Reply::json(
            200,
            r#"{"runId":"job id","cost":0.21,"paymentSource":"fixture source","isApiRequest":true}"#,
        );
        submit.headers = "x-nanogpt-pricing: {\"costUsd\":0.9}\r\n";
        let refusal = Reply {
            length: Some(100),
            ..Reply::json(status, "bad")
        };
        let later = Reply::json(
            200,
            r#"{"status":"completed","url":"https://fixture.invalid/audio.mp3"}"#,
        );
        let requests = with_provider(vec![submit, refusal, later], |ctx| {
            let error = node_run("tts", ctx).unwrap_err();
            let message = if status == 402 {
                "out of balance"
            } else {
                "API key rejected"
            };
            assert!(error.message.contains(message), "{}", error.message);
            let result = error.result.unwrap();
            assert_eq!(result.nodes["work"].cost, Some(0.21));
            assert_eq!(result.cost_usd, 0.21);
        });
        assert_eq!(requests, [
            "POST /api/v1/audio/speech",
            "GET /api/tts/status?runId=job+id&model=test&cost=0.21&paymentSource=fixture+source&isApiRequest=true",
        ]);
    }
}

#[test]
fn video_3d_and_lipsync_keep_submit_cost_when_poll_auth_is_rejected() {
    let inputs = std::collections::HashMap::from([
        (
            "image".into(),
            json!("https://fixture.invalid/portrait.png"),
        ),
        ("audio".into(), json!("https://fixture.invalid/audio.mp3")),
    ]);
    for node_type in ["tvideo", "model3d", "lipsync"] {
        let graph = materialize(&json!({
            "nodes": [{"id": "work", "type": node_type, "fields": {"model": "test", "prompt": "hello"}}]
        })).unwrap();
        let submit = Reply {
            headers: "x-nanogpt-pricing: {\"costUsd\":0.21}\r\n",
            ..Reply::json(200, r#"{"runId":"job"}"#)
        };
        let refusal = Reply {
            length: Some(100),
            ..Reply::json(403, "bad")
        };
        let later = Reply::json(
            200,
            r#"{"status":"completed","output":{"url":"https://fixture.invalid/video.mp4"}}"#,
        );
        let requests = with_provider(vec![submit, refusal, later], |ctx| {
            let error = run_node(&graph.nodes[0], &inputs, ctx)
                .err()
                .expect("polling should fail");
            assert!(
                error.message.contains("API key rejected"),
                "{node_type}: {}",
                error.message
            );
            assert_eq!(error.cost, Some(0.21));
        });
        assert_eq!(
            requests,
            [
                "POST /api/generate-video",
                "GET /api/video/status?requestId=job"
            ]
        );
    }
}

#[test]
fn lipsync_uses_the_catalog_to_choose_whether_to_send_a_portrait() {
    let model = "fixture-avatar";
    let audio = "https://fixture.invalid/line.wav";
    let image = "https://fixture.invalid/portrait.png";
    let mut data = json!({
        "nodes": [
            {"id":"audio","type":"aupload","fields":{"audio":audio}},
            {"id":"avatar","type":"lipsync","fields":{"model":model}},
        ],
        "links":[{"from":{"node":"audio","port":"audio"},"to":{"node":"avatar","port":"audio"}}],
    });
    let audio_only = materialize(&data).unwrap();
    data["nodes"].as_array_mut().unwrap().push(json!({"id":"image","type":"upload","fields":{"image":image}}));
    data["links"].as_array_mut().unwrap().push(json!({
        "from":{"node":"image","port":"image"},"to":{"node":"avatar","port":"image"},
    }));
    let with_portrait = materialize(&data).unwrap();

    let preset = json!({"video":[{"id":model,"capabilities":{"image_to_video":false}}]});
    let normalized = json!({"video":[{"id":model,"i2v":false}]});
    let image_required = json!({"video":[{"id":model,"capabilities":{"image_to_video":true}}]});
    for (graph, catalog, portrait) in [
        (&audio_only, &preset, None),
        (&with_portrait, &preset, None),
        (&audio_only, &normalized, None),
        (&with_portrait, &image_required, Some(image)),
    ] {
        let submit = Reply {
            headers: "x-nanogpt-pricing: {\"costUsd\":0.4}\r\n",
            ..Reply::json(200, r#"{"runId":"job"}"#)
        };
        let status = Reply::json(200, r#"{"status":"completed","output":{"url":"https://fixture.invalid/video.mp4"}}"#);
        let requests = with_provider_requests(vec![submit, status], |ctx| {
            let mut configured = Ctx::new(&ctx.base_url, &ctx.api_key);
            configured.catalog = Some(catalog.clone());
            let result = run(graph, &configured).unwrap_or_else(|error| panic!("{}", error.message));
            assert_eq!(result.outputs["avatar"], "https://fixture.invalid/video.mp4");
            assert_eq!(result.cost_usd, 0.4);
        });
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].0, "POST /api/generate-video");
        assert_eq!(requests[1].0, "GET /api/video/status?requestId=job");
        let payload: Value = serde_json::from_slice(&requests[0].1).unwrap();
        assert_eq!(payload["audioUrl"], audio);
        assert_eq!(payload.get("imageDataUrl").and_then(Value::as_str), portrait);
    }

    for catalog in [None, Some(json!({"video":[]})), Some(image_required)] {
        let requests = with_provider(vec![Reply::json(400, "unexpected request")], |ctx| {
            let mut configured = Ctx::new(&ctx.base_url, &ctx.api_key);
            configured.catalog = catalog;
            let failure = run(&audio_only, &configured).unwrap_err();
            assert_eq!(failure.result.unwrap().nodes["avatar"].error.as_deref(), Some("no image input"));
        });
        assert!(requests.is_empty(), "{requests:?}");
    }
}

#[test]
fn transient_audio_polls_retry_without_counting_poll_cost() {
    let replies = vec![
        Reply {
            headers: "x-nanogpt-pricing: {\"costUsd\":0.21}\r\n",
            ..Reply::json(200, r#"{"runId":"job"}"#)
        },
        Reply::json(503, "insufficient funds"),
        Reply::json(429, "try again"),
        Reply::json(408, "try again"),
        Reply::json(200, "{"),
        Reply::json(
            200,
            r#"{"status":"completed","url":"https://fixture.invalid/audio.mp3","cost":0.21}"#,
        ),
    ];
    let requests = with_provider(replies, |ctx| {
        let result = node_run("tts", ctx).unwrap_or_else(|error| panic!("{}", error.message));
        assert_eq!(result.outputs["work"], "https://fixture.invalid/audio.mp3");
        assert_eq!(result.nodes["work"].cost, Some(0.21));
        assert_eq!(result.cost_usd, 0.21);
    });
    assert_eq!(requests.len(), 6);
}

#[test]
fn poll_limit_clamps_sleep_and_rejects_a_late_success() {
    let start = Instant::now();
    let err = poll_loop::<()>(
        Duration::from_secs(2),
        Duration::from_millis(10),
        "timeout".into(),
        || panic!("the initial wait consumes the poll budget"),
    )
    .unwrap_err();
    assert_eq!(err.0, "timeout");
    assert!(start.elapsed() < Duration::from_secs(1));

    let mut ran_step = false;
    let err = poll_loop(
        Duration::ZERO,
        Duration::from_millis(150),
        "timeout".into(),
        || {
            ran_step = true;
            thread::sleep(Duration::from_millis(175));
            Ok(PollStep::Done("late"))
        },
    )
    .unwrap_err();
    assert!(ran_step);
    assert_eq!(err.0, "timeout");
}

#[test]
fn poll_limit_bounds_the_blocking_response_body_read() {
    let reply = Reply {
        length: Some(200),
        stall: true,
        ..Reply::json(200, "{")
    };
    let requests = with_provider(vec![reply], |ctx| {
        let start = Instant::now();
        let err = ctx
            .poll_json::<()>(
                "/status",
                Duration::ZERO,
                Duration::from_millis(150),
                "timeout".into(),
                |_| panic!("incomplete body reached the classifier"),
            )
            .unwrap_err();
        assert_eq!(err.0, "timeout");
        assert!(start.elapsed() < Duration::from_secs(1));
    });
    assert_eq!(requests, ["GET /status"]);
}
