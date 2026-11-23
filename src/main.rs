//! nanoodle-rs CLI:
//!   nanoodle-rs load <graph.json>  → load-only report (JSON)
//!   nanoodle-rs run  <graph.json> [--base-url <url>] [--api-key <key>]
//!                                  [--catalog <file.json>]
//!                                  → run report (JSON)
//! Output shapes match the conformance suite's runner protocol.

use nanoodle::{materialize, run, Ctx, Graph};
use serde_json::{json, Map, Value};
use std::process::ExitCode;

fn load_graph(path: &str) -> Result<(Value, Graph), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
    let data: Value = serde_json::from_str(&text).map_err(|e| format!("invalid JSON: {}", e))?;
    let g = materialize(&data).map_err(|e| e.0)?;
    Ok((data, g))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (mode, path) = match (args.get(1), args.get(2)) {
        (Some(m), Some(p)) if m == "load" || m == "run" => (m.clone(), p.clone()),
        _ => {
            eprintln!("usage: nanoodle-rs <load|run> <graph.json>");
            return ExitCode::from(2);
        }
    };
    let (_data, graph) = match load_graph(&path) {
        Ok(v) => v,
        Err(msg) => {
            println!("{}", json!({ "loadError": msg }).to_string());
            return ExitCode::FAILURE;
        }
    };
    let warnings: Vec<Value> = graph.warnings.iter().map(|w| json!(w)).collect();
    if mode == "load" {
        let mut types: Vec<&str> = graph.nodes.iter().map(|n| n.ty.as_str()).collect();
        types.sort();
        println!(
            "{}",
            json!({
                "loadWarnings": warnings,
                "loadOnly": {
                    "nodeCount": graph.nodes.len(),
                    "types": types,
                    "nWarnings": graph.warnings.len(),
                }
            })
            .to_string()
        );
        return ExitCode::SUCCESS;
    }
    let mut base_url = "https://nano-gpt.com".to_string();
    let mut api_key = String::new();
    let mut catalog_path: Option<String> = None;
    let mut it = args.iter().skip(3);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--base-url" => base_url = it.next().cloned().unwrap_or_default(),
            "--api-key" => api_key = it.next().cloned().unwrap_or_default(),
            "--catalog" => catalog_path = it.next().cloned(),
            _ => {}
        }
    }
    let mut ctx = Ctx::new(&base_url, &api_key);
    if let Some(p) = catalog_path {
        if let Ok(text) = std::fs::read_to_string(&p) {
            ctx.catalog = serde_json::from_str(&text).ok();
        }
    }
    match run(&graph, &ctx) {
        Ok(result) => {
            let mut nodes = Map::new();
            for n in &graph.nodes {
                if let Some(rec) = result.nodes.get(&n.id) {
                    nodes.insert(
                        n.id.clone(),
                        json!({ "status": rec.status, "error": rec.error }),
                    );
                }
            }
            println!(
                "{}",
                json!({
                    "loadWarnings": warnings,
                    "outputs": result.outputs,
                    "nodes": nodes,
                    "cost": result.cost_usd,
                    "errors": result.errors,
                })
                .to_string()
            );
            ExitCode::SUCCESS
        }
        Err(f) => {
            // A failed run carries its partial result (JS RunError.result):
            // emit the records/outputs/cost alongside the error.
            let mut nodes = Map::new();
            let mut outputs = json!({});
            let mut cost = json!(null);
            let mut errors = json!([]);
            if let Some(result) = &f.result {
                for n in &graph.nodes {
                    if let Some(rec) = result.nodes.get(&n.id) {
                        nodes.insert(
                            n.id.clone(),
                            json!({ "status": rec.status, "error": rec.error }),
                        );
                    }
                }
                outputs = json!(result.outputs);
                cost = json!(result.cost_usd);
                errors = json!(result.errors);
            }
            println!(
                "{}",
                json!({
                    "loadWarnings": warnings,
                    "runError": f.message,
                    "outputs": outputs,
                    "nodes": nodes,
                    "cost": cost,
                    "errors": errors,
                })
                .to_string()
            );
            ExitCode::FAILURE
        }
    }
}
