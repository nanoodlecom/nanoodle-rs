//! Drive one local Chromium process through its loopback DevTools endpoint.

use super::{err, TIMEOUT_MESSAGE};
use crate::{NanoodleError, Result};
use serde_json::{json, Value};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tungstenite::{client, protocol::WebSocketConfig, Message, WebSocket};

const MAX_MESSAGE: usize = 32 * 1024 * 1024;
const STARTUP_LIMIT: Duration = Duration::from_secs(30);
const CLOSE_WAIT: Duration = Duration::from_millis(300);

fn remaining(deadline: Instant) -> io::Result<Duration> {
    let time = deadline.saturating_duration_since(Instant::now());
    if time.is_zero() {
        return Err(io::Error::from(io::ErrorKind::TimedOut));
    }
    Ok(time)
}

// Apply the absolute deadline to each socket read/write, including a Chrome
// response split across several TCP packets or WebSocket frames.
struct DeadlineSocket {
    stream: TcpStream,
    deadline: Instant,
}

impl Read for DeadlineSocket {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.stream
            .set_read_timeout(Some(remaining(self.deadline)?))?;
        self.stream.read(buf)
    }
}

impl Write for DeadlineSocket {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.stream
            .set_write_timeout(Some(remaining(self.deadline)?))?;
        self.stream.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stream.flush()
    }
}

struct Cdp {
    socket: WebSocket<DeadlineSocket>,
    id: u64,
}

impl Cdp {
    fn transport_error(&self, error: tungstenite::Error) -> NanoodleError {
        if Instant::now() >= self.socket.get_ref().deadline {
            err(TIMEOUT_MESSAGE)
        } else {
            err(format!("render3d: Chromium connection failed: {error}"))
        }
    }

    fn send(&mut self, method: &str, params: Value, session: Option<&str>) -> Result<Value> {
        remaining(self.socket.get_ref().deadline).map_err(|_| err(TIMEOUT_MESSAGE))?;
        self.id += 1;
        let mut request = json!({"id": self.id, "method": method, "params": params});
        if let Some(session) = session {
            request["sessionId"] = json!(session);
        }
        self.socket
            .send(Message::text(request.to_string()))
            .map_err(|error| self.transport_error(error))?;
        loop {
            remaining(self.socket.get_ref().deadline).map_err(|_| err(TIMEOUT_MESSAGE))?;
            let message = self
                .socket
                .read()
                .map_err(|error| self.transport_error(error))?;
            let Message::Text(text) = message else {
                continue;
            };
            let mut response: Value = serde_json::from_str(&text)
                .map_err(|error| err(format!("render3d: invalid Chromium response: {error}")))?;
            if response["id"].as_u64() != Some(self.id) {
                continue;
            }
            if let Some(error) = response.pointer("/error/message").and_then(Value::as_str) {
                return Err(err(format!("render3d: {method}: {error}")));
            }
            remaining(self.socket.get_ref().deadline).map_err(|_| err(TIMEOUT_MESSAGE))?;
            return Ok(response["result"].take());
        }
    }

    fn initial_target(&mut self) -> Result<String> {
        loop {
            let targets = self.send("Target.getTargets", json!({}), None)?;
            let page = targets["targetInfos"].as_array().and_then(|entries| {
                entries
                    .iter()
                    .find(|t| t["type"] == "page" && t["url"] == "about:blank")
            });
            if let Some(id) = page.and_then(|target| target["targetId"].as_str()) {
                return Ok(id.to_string());
            }
            let time = remaining(self.socket.get_ref().deadline)
                .map_err(|_| err("render3d: Chromium did not open its initial page"))?;
            std::thread::sleep(Duration::from_millis(20).min(time));
        }
    }
}

pub(super) struct Chromium {
    child: Child,
    cdp: Option<Cdp>,
    session: String,
}

impl Chromium {
    pub(super) fn open(exe: &Path, profile: &Path, port: u16, deadline: Instant) -> Result<Self> {
        remaining(deadline).map_err(|_| err(TIMEOUT_MESSAGE))?;
        let startup_deadline = deadline.min(Instant::now() + STARTUP_LIMIT);
        let mut profile_arg = std::ffi::OsString::from("--user-data-dir=");
        profile_arg.push(profile);
        let child = Command::new(exe)
            .args([
                "--headless=new",
                "--remote-debugging-port=0",
                "--remote-debugging-address=127.0.0.1",
                "--no-sandbox",
                "--incognito",
                "--no-first-run",
                "--no-proxy-server",
                "--disable-background-networking",
                "--disable-dev-shm-usage",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
            ])
            .arg(profile_arg)
            .arg("about:blank")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| err(format!("render3d: could not launch Chromium: {error}")))?;
        let mut browser = Self {
            child,
            cdp: None,
            session: String::new(),
        };
        let (address, path) = browser.endpoint(profile, startup_deadline, deadline)?;
        let time = remaining(startup_deadline)
            .map_err(|_| err("render3d: Chromium did not start in time"))?;
        let stream = TcpStream::connect_timeout(&address, time)
            .map_err(|error| err(format!("render3d: could not reach Chromium: {error}")))?;
        let config = WebSocketConfig::default()
            .max_message_size(Some(MAX_MESSAGE))
            .max_frame_size(Some(MAX_MESSAGE));
        let (socket, _) = client::client_with_config(
            format!("ws://{address}{path}"),
            DeadlineSocket {
                stream,
                deadline: startup_deadline,
            },
            Some(config),
        )
        .map_err(|error| err(format!("render3d: could not connect to Chromium: {error}")))?;
        browser.cdp = Some(Cdp { socket, id: 0 });
        let cdp = browser.cdp.as_mut().expect("Chromium connected");
        // A newly created target can belong to Chromium's dormant regular
        // context. Navigate the incognito page started by our command line.
        let target_id = cdp.initial_target()?;
        let attached = cdp.send(
            "Target.attachToTarget",
            json!({"targetId": target_id, "flatten": true}),
            None,
        )?;
        browser.session = attached["sessionId"]
            .as_str()
            .ok_or_else(|| err("render3d: Chromium did not attach to the target"))?
            .into();
        cdp.socket.get_mut().deadline = deadline;
        let url = format!("http://127.0.0.1:{port}/scene.html");
        let navigation = browser.command("Page.navigate", json!({"url": url}))?;
        if navigation["isDownload"] == true {
            return Err(err(
                "render3d: scene navigation started an unexpected download",
            ));
        }
        if let Some(error) = navigation["errorText"].as_str() {
            return Err(err(format!("render3d: scene navigation failed: {error}")));
        }
        browser.wait_for_scene(&url, deadline)?;
        Ok(browser)
    }

    fn endpoint(
        &mut self,
        profile: &Path,
        startup_deadline: Instant,
        deadline: Instant,
    ) -> Result<(SocketAddr, String)> {
        let active_port = profile.join("DevToolsActivePort");
        loop {
            if let Ok(file) = std::fs::read_to_string(&active_port) {
                let mut lines = file.lines();
                let port = lines
                    .next()
                    .and_then(|line| line.trim().parse::<u16>().ok());
                let path = lines.next().map(str::trim);
                if let (Some(port), Some(path)) = (port, path) {
                    if port != 0
                        && path.strip_prefix("/devtools/browser/").is_some_and(|id| {
                            !id.is_empty()
                                && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
                        })
                    {
                        return Ok((SocketAddr::from(([127, 0, 0, 1], port)), path.to_string()));
                    }
                }
            }
            if self
                .child
                .try_wait()
                .map_err(|error| err(format!("render3d: {error}")))?
                .is_some()
            {
                return Err(err("render3d: Chromium exited during startup"));
            }
            let time = remaining(startup_deadline).map_err(|_| {
                if Instant::now() >= deadline {
                    err(TIMEOUT_MESSAGE)
                } else {
                    err("render3d: Chromium did not start in time")
                }
            })?;
            std::thread::sleep(Duration::from_millis(20).min(time));
        }
    }

    fn command(&mut self, method: &str, params: Value) -> Result<Value> {
        self.cdp
            .as_mut()
            .expect("Chromium connected")
            .send(method, params, Some(&self.session))
    }

    fn evaluate(&mut self, expression: &str) -> Result<Value> {
        let mut response = self.command(
            "Runtime.evaluate",
            json!({"expression": expression, "returnByValue": true}),
        )?;
        if let Some(error) = response.get("exceptionDetails") {
            let detail = error
                .pointer("/exception/description")
                .and_then(Value::as_str)
                .or_else(|| error["text"].as_str())
                .unwrap_or("unknown scene error");
            return Err(err(format!(
                "render3d: scene did not produce a frame: {detail}"
            )));
        }
        Ok(response
            .pointer_mut("/result/value")
            .map(Value::take)
            .unwrap_or(Value::Null))
    }

    fn wait_for_scene(&mut self, url: &str, deadline: Instant) -> Result<()> {
        // Each probe is synchronous, so a probe on the initial about:blank
        // context cannot wait forever or be mistaken for the navigated scene.
        let probe = "({url: location.href, loaded: document.readyState === 'complete', initialized: !!window.__render3d, ready: !!window.__render3d?.ready, error: window.__render3d?.error})";
        loop {
            match self.evaluate(probe) {
                Ok(state) if state["url"].as_str() == Some(url) => {
                    if let Some(error) = state["error"].as_str() {
                        return Err(err(format!(
                            "render3d: scene did not produce a frame: {error}"
                        )));
                    }
                    if state["loaded"] == true && state["initialized"] == false {
                        return Err(err("render3d: scene did not initialize"));
                    }
                    if state["loaded"] == true && state["ready"] == true {
                        return Ok(());
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    let text = error.0.to_ascii_lowercase();
                    if !text.contains("execution context was destroyed")
                        && !text.contains("cannot find context")
                        && !text.contains("cannot find default execution context")
                    {
                        return Err(error);
                    }
                }
            }
            let time = remaining(deadline).map_err(|_| err(TIMEOUT_MESSAGE))?;
            std::thread::sleep(Duration::from_millis(30).min(time));
        }
    }

    pub(super) fn frame(&mut self, fields: &Value) -> Result<String> {
        let result = self.evaluate(&format!(
            "(() => {{ window.__render3d.apply({fields}); return document.getElementById('c').toDataURL('image/png'); }})()"
        ))?;
        match result {
            Value::String(data) => Ok(data),
            _ => Err(err("render3d: scene did not produce a frame")),
        }
    }
}

impl Drop for Chromium {
    fn drop(&mut self) {
        if let Some(cdp) = &mut self.cdp {
            let limit = Instant::now() + CLOSE_WAIT;
            cdp.socket.get_mut().deadline = limit;
            let _ = cdp
                .socket
                .send(Message::text(r#"{"id":0,"method":"Browser.close"}"#));
            loop {
                match self.child.try_wait() {
                    Ok(Some(_)) => return,
                    Ok(None) if Instant::now() < limit => {
                        std::thread::sleep(Duration::from_millis(10))
                    }
                    _ => break,
                }
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
