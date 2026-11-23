use nanoodle::{materialize, run, Ctx, Graph};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, VecDeque},
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{mpsc, Arc, Mutex},
    thread,
    time::Duration,
};

struct Script {
    answers: VecDeque<Value>,
    calls: Vec<String>,
}

fn serve(mut stream: TcpStream, script: &Mutex<Script>, models: &HashMap<String, String>) {
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let mut reader = BufReader::new(&mut stream);
    let mut request = String::new();
    reader.read_line(&mut request).unwrap();
    let path = request.split_whitespace().nth(1).unwrap_or("");
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
    let mut body = vec![0; size];
    reader.read_exact(&mut body).unwrap();
    let payload: Value = serde_json::from_slice(&body).unwrap();
    let mut script = script.lock().unwrap();
    let response = match path {
        "/api/v1/decisions" => {
            let id = if script.calls.iter().any(|id| id == "n2") {
                "n5"
            } else {
                "n2"
            };
            script.calls.push(id.to_string());
            let noul = script.answers.pop_front().unwrap();
            let answer = if noul.is_null() {
                Value::Null
            } else {
                json!({"type":"noul", "noul":noul})
            };
            json!({"answers":{"answer":answer}, "usage":{"cost":0.001}})
        }
        "/api/v1/chat/completions" => {
            let id = &models[payload["model"].as_str().unwrap()];
            script.calls.push(id.clone());
            json!({"choices":[{"message":{"content":id}}], "cost":0.001})
        }
        _ => panic!("unexpected request to {path}"),
    }
    .to_string();
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        response.len(),
        response,
    )
    .unwrap();
}

struct StopServer(mpsc::Sender<()>);

impl Drop for StopServer {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

fn with_decision_provider(graph: &Graph, check: impl FnOnce(&Ctx, &Mutex<Script>)) {
    let models = graph
        .nodes
        .iter()
        .filter(|n| n.ty == "llm")
        .map(|n| (n.field_str("model").unwrap(), n.id.clone()))
        .collect();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let ctx = Ctx::new(&base, "offline-fixture");
    let script = Arc::new(Mutex::new(Script {
        answers: VecDeque::new(),
        calls: Vec::new(),
    }));
    thread::scope(|s| {
        let (send, recv) = mpsc::channel();
        let _stop = StopServer(send);
        let server_script = Arc::clone(&script);
        s.spawn(move || loop {
            match listener.accept() {
                Ok((stream, _)) => serve(stream, &server_script, &models),
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    if recv.recv_timeout(Duration::from_millis(5)).is_ok() {
                        break;
                    }
                }
                Err(err) => panic!("{err}"),
            }
        });
        check(&ctx, &script);
    });
}

#[test]
fn shared_branch_fixture_and_legacy_round_trip() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/decider-branches.json")).unwrap();
    let graph = materialize(&fixture).unwrap();
    with_decision_provider(&graph, |ctx, script| {
        for scenario in fixture["cases"].as_array().unwrap() {
            let name = scenario["name"].as_str().unwrap();
            {
                let mut state = script.lock().unwrap();
                state.calls.clear();
                state.answers = scenario["answers"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .cloned()
                    .collect();
            }
            let result = match run(&graph, ctx) {
                Ok(result) => {
                    assert_ne!(scenario["error"], true, "{name}");
                    assert_eq!(result.outputs["Answer"], scenario["output"], "{name}");
                    for id in scenario["skipped"].as_array().unwrap() {
                        let rec = &result.nodes[id.as_str().unwrap()];
                        assert_eq!(rec.status, "skipped", "{name}");
                        assert!(rec.cost.is_none(), "{name}");
                    }
                    result
                }
                Err(error) => {
                    assert_eq!(scenario["error"], true, "{name}: {}", error.message);
                    let result = error.result.unwrap();
                    assert!(result
                        .errors
                        .iter()
                        .any(|e| e.contains("valid yes/no answer")));
                    result
                }
            };
            let state = script.lock().unwrap();
            let mut actual = state.calls.clone();
            actual.sort();
            let mut expected: Vec<_> = scenario["calls"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            expected.sort_unstable();
            assert_eq!(actual, expected, "{name}");
            assert!(state.answers.is_empty(), "{name}");
            assert!(
                (result.cost_usd - 0.001 * expected.len() as f64).abs() < 1e-9,
                "{name}"
            );
        }
        for branching in [false, true] {
            {
                let mut state = script.lock().unwrap();
                state.calls.clear();
                state.answers = [json!(0)].into();
            }
            let mut graph = fixture["legacy"].clone();
            if branching {
                graph["nodes"][1]["type"] = json!("decide-branch");
                graph["links"][1]["from"]["port"] = json!("yes");
            }
            graph["nodes"].as_array_mut().unwrap().push(json!({
                "id": "failed", "type": "image",
                "fields": {"prompt": "draw a box", "onError": "skip"}
            }));
            let result = run(&materialize(&graph).unwrap(), ctx)
                .unwrap_or_else(|e| panic!("{}", e.message));
            assert_eq!(result.nodes["n2"].status, if branching { "done" } else { "gated" });
            assert_eq!(result.nodes["n4"].status, "skipped");
            assert!(result.nodes["n4"].error.is_none());
            assert_eq!(result.nodes["failed"].status, "skipped");
            assert!(result.nodes["failed"].error.is_some());
            assert!(result.outputs.is_empty());
            assert_eq!(result.cost_usd, 0.001);
            assert_eq!(script.lock().unwrap().calls, ["n2"]);
        }
    });
}

fn wire(from: &str, from_port: &str, to: &str, to_port: &str) -> Value {
    json!({"from":{"node":from,"port":from_port},"to":{"node":to,"port":to_port}})
}

#[test]
fn mutually_exclusive_upstream_routes_can_share_a_destination() {
    let graph = materialize(&json!({
        "nodes": [
            {"id":"n2","type":"decide-branch","fields":{"model":"fixture","question":"Is this yes?"}},
            {"id":"left","type":"text"},
            {"id":"right","type":"text"},
            {"id":"n4","type":"llm","fields":{"model":"consumer"}},
        ],
        "links": [
            wire("n2", "yes", "left", "text"),
            wire("n2", "no", "right", "text"),
            wire("left", "text", "n4", "prompt"),
            wire("right", "text", "n4", "prompt"),
        ],
    })).unwrap();
    assert!(graph.warnings.is_empty());
    with_decision_provider(&graph, |ctx, script| {
        for (answer, active, inactive) in [(1, "left", "right"), (0, "right", "left")] {
            {
                let mut script = script.lock().unwrap();
                script.calls.clear();
                script.answers = [json!(answer)].into();
            }
            let result = run(&graph, ctx).unwrap_or_else(|e| panic!("{}", e.message));
            assert_eq!(result.nodes[active].status, "done");
            assert_eq!(result.nodes[inactive].status, "skipped");
            assert_eq!(result.nodes["n4"].status, "done");
            assert_eq!(script.lock().unwrap().calls, ["n2", "n4"]);
        }
    });
}

#[test]
fn branch_or_legacy_gate_checks_live_conflicts_before_calling_the_consumer() {
    for (gate_ty, port) in [("decide-branch", "yes"), ("decide", "text")] {
        let graph = materialize(&json!({
            "nodes": [
                {"id":"n2","type":gate_ty,"fields":{
                    "model":"fixture","question":"Is this yes?","mode":"yesno","gate":true
                }},
                {"id":"left","type":"text","fields":{"text":"first"}},
                {"id":"right","type":"text","fields":{"text":"second"}},
                {"id":"n4","type":"llm","fields":{"model":"consumer"}},
            ],
            "links": [
                wire("n2", port, "n4", "system"),
                wire("left", "text", "n4", "prompt"),
                wire("right", "text", "n4", "prompt"),
            ],
        })).unwrap();
        assert!(graph.warnings.is_empty(), "{gate_ty}: {:?}", graph.warnings);
        with_decision_provider(&graph, |ctx, script| {
            for answer in [0, 1] {
                {
                    let mut script = script.lock().unwrap();
                    script.calls.clear();
                    script.answers = [json!(answer)].into();
                }
                let result = if answer == 1 {
                    let failure = run(&graph, ctx).unwrap_err();
                    let diagnostic = "input n4.prompt has competing wires from left.text, right.text. Leave one active wire for this input.";
                    let result = failure.result.unwrap();
                    assert_eq!(result.nodes["n4"].error.as_deref(), Some(diagnostic));
                    result
                } else {
                    let result = run(&graph, ctx).unwrap_or_else(|e| panic!("{}", e.message));
                    assert_eq!(result.nodes["n4"].status, "skipped");
                    result
                };
                assert_eq!(result.cost_usd, 0.001);
                assert_eq!(script.lock().unwrap().calls, ["n2"]);
            }
        });
    }
}
