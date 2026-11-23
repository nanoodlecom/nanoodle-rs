//! nanoodle-rs CLI:
//!   nanoodle-rs load <graph.json>  → load-only report (JSON)
//!   nanoodle-rs run  <graph.json> [--base-url <url>] [--api-key <key>]
//!                                  [--catalog <file.json>]
//!                                  → run report (JSON)
//! Output shapes match the conformance suite's runner protocol.

use nanoodle::{materialize, run, Ctx, Graph};
use serde_json::{json, Map, Value};
use std::process::ExitCode;

const USAGE: &str = "usage: nanoodle-rs load <graph.json>\n       nanoodle-rs run <graph.json> [--base-url <url>] [--api-key <key>] [--catalog <file.json>]";

fn usage_error(message: &str) -> ExitCode {
    eprintln!("{message}\n{USAGE}");
    ExitCode::from(2)
}

fn positional_value(value: &str) -> bool {
    !value.starts_with('-') && !value.trim().is_empty()
}

fn read_catalog(path: &str) -> Result<Value, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("cannot read --catalog {path:?}: {err}"))?;
    let catalog: Value = serde_json::from_str(&text)
        .map_err(|err| format!("invalid JSON in --catalog {path:?}: {err}"))?;
    let sections = catalog
        .as_object()
        .ok_or_else(|| format!("invalid --catalog {path:?}: expected a JSON object"))?;
    for name in ["chat", "image", "video", "audio", "model3d"] {
        if sections
            .get(name)
            .is_some_and(|section| !section.is_array())
        {
            return Err(format!(
                "invalid --catalog {path:?}: catalog section '{name}' must be an array"
            ));
        }
    }
    Ok(catalog)
}

fn run_context(options: &[String]) -> Result<Ctx, String> {
    let mut base_url = None;
    let mut api_key = None;
    let mut catalog_path = None;
    let mut args = options.iter().map(String::as_str);
    while let Some(option) = args.next() {
        let target = match option {
            "--base-url" => &mut base_url,
            "--api-key" => &mut api_key,
            "--catalog" => &mut catalog_path,
            _ if option.starts_with('-') => {
                let name = option.split_once('=').map_or(option, |(name, _)| name);
                return Err(format!("unknown option {name}"));
            }
            _ => return Err("unexpected argument after the graph path".into()),
        };
        let value = args
            .next()
            .filter(|value| positional_value(value))
            .ok_or_else(|| format!("{option} requires a value"))?;
        *target = Some(value);
    }
    let catalog = catalog_path.map(read_catalog).transpose()?;
    let mut ctx = Ctx::new(
        base_url.unwrap_or("https://nano-gpt.com"),
        api_key.unwrap_or(""),
    );
    ctx.catalog = catalog;
    Ok(ctx)
}

fn load_graph(path: &str) -> Result<(Value, Graph), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
    let data: Value = serde_json::from_str(&text).map_err(|e| format!("invalid JSON: {}", e))?;
    let g = materialize(&data).map_err(|e| e.0)?;
    Ok((data, g))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mode, path, options @ ..] = args.as_slice() else {
        return usage_error("expected a command and graph file");
    };
    if !positional_value(path) {
        return usage_error("expected a graph file after the command");
    }
    let ctx = match mode.as_str() {
        "load" if options.is_empty() => None,
        "load" => return usage_error("load takes no options"),
        "run" => match run_context(options) {
            Ok(ctx) => Some(ctx),
            Err(message) => return usage_error(&message),
        },
        _ => return usage_error("expected the load or run command"),
    };
    let (_data, graph) = match load_graph(path) {
        Ok(v) => v,
        Err(msg) => {
            println!("{}", json!({ "loadError": msg }).to_string());
            return ExitCode::FAILURE;
        }
    };
    let warnings: Vec<Value> = graph.warnings.iter().map(|w| json!(w)).collect();
    let Some(ctx) = ctx else {
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
    };
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
