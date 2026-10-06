// Copyright 2026 The Sashiko Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Tools served by remote Model Context Protocol servers.
//!
//! A minimal client for the Streamable HTTP transport: `initialize`,
//! `tools/list` and `tools/call`, with JSON or SSE responses. Each allowed
//! server tool becomes an [`McpTool`] registered next to the built-in tools.
//! See designs/DESIGN_MCP_CLIENT_TOOLS.md.

use crate::settings::{McpServerSettings, McpSettings};
use crate::toolbox::SashikoToolContext;
use crate::toolbox::framework::LlmTool;
use crate::utils::redact_secret;
use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, RwLock};
use tracing::{info, warn};

/// The protocol revision offered in `initialize`. Servers answer with the
/// revision they speak, which is sent back on every later request.
const PROTOCOL_VERSION: &str = "2025-06-18";
/// Cap on one HTTP response body, independent of the per-tool output cap.
const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
/// Cap on `tools/list` pages followed, so a server cannot loop discovery.
const MAX_LIST_PAGES: usize = 10;
/// Cap on a tool description taken from the server.
const MAX_DESCRIPTION_BYTES: usize = 2048;
/// Longest tool name providers accept.
const MAX_TOOL_NAME_BYTES: usize = 64;
const SESSION_HEADER: &str = "mcp-session-id";
const PROTOCOL_HEADER: &str = "mcp-protocol-version";

/// Connection state for one server.
#[derive(Clone)]
struct Session {
    id: Option<String>,
    protocol_version: String,
    /// Bumped by every successful handshake, so a caller that saw a session
    /// expire can tell whether someone else has already replaced it.
    generation: u64,
}

/// A JSON-RPC client for one MCP server.
pub struct McpClient {
    name: String,
    url: String,
    http: reqwest::Client,
    token: Option<String>,
    /// The current session. Requests copy it and release the lock before
    /// any network I/O.
    session: RwLock<Session>,
    /// Held for the whole handshake, so concurrent callers that see the
    /// session expire reconnect once rather than racing.
    reconnect: Mutex<()>,
    next_id: AtomicU64,
}

impl McpClient {
    fn new(settings: &McpServerSettings, token: Option<String>) -> Result<Self> {
        // No redirects: the token must only ever reach the configured origin.
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(settings.timeout_secs))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self {
            name: settings.name.clone(),
            url: settings.url.clone(),
            http,
            token,
            session: RwLock::new(Session {
                id: None,
                protocol_version: PROTOCOL_VERSION.to_string(),
                generation: 0,
            }),
            reconnect: Mutex::new(()),
            next_id: AtomicU64::new(1),
        })
    }

    /// Runs the `initialize` handshake and records the session it opens.
    async fn initialize(&self) -> Result<()> {
        let seen = self.session.read().await.generation;
        self.reconnect(seen).await
    }

    /// Opens a new session unless one newer than generation `seen` already
    /// exists. Requests in flight keep using the old session until the new
    /// one is recorded; nothing is cleared in between.
    async fn reconnect(&self, seen: u64) -> Result<()> {
        let _one_at_a_time = self.reconnect.lock().await;
        if self.session.read().await.generation != seen {
            return Ok(());
        }
        let fresh = Session {
            id: None,
            protocol_version: PROTOCOL_VERSION.to_string(),
            generation: seen,
        };
        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": { "name": "sashiko", "version": env!("CARGO_PKG_VERSION") }
        });
        let (result, session_id) = self.post_request("initialize", params, &fresh).await?;
        let version = result["protocolVersion"]
            .as_str()
            .unwrap_or(PROTOCOL_VERSION)
            .to_string();
        let session = Session {
            id: session_id,
            protocol_version: version,
            generation: seen + 1,
        };
        self.post_notification("notifications/initialized", &session)
            .await?;
        *self.session.write().await = session;
        Ok(())
    }

    /// Sends a request, reconnecting once if the server dropped the session.
    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let session = self.session.read().await.clone();
        match self.post_request(method, params.clone(), &session).await {
            Err(e) if e.downcast_ref::<SessionExpired>().is_some() => {
                info!("MCP server {}: session expired, reconnecting", self.name);
                self.reconnect(session.generation).await?;
                let session = self.session.read().await.clone();
                Ok(self.post_request(method, params, &session).await?.0)
            }
            other => Ok(other?.0),
        }
    }

    fn post(&self, body: &Value, session: &Session) -> reqwest::RequestBuilder {
        let mut req = self
            .http
            .post(&self.url)
            .header(
                reqwest::header::ACCEPT,
                "application/json, text/event-stream",
            )
            .header(PROTOCOL_HEADER, &session.protocol_version)
            .json(body);
        if let Some(id) = &session.id {
            req = req.header(SESSION_HEADER, id);
        }
        if let Some(token) = &self.token {
            req = req.bearer_auth(token);
        }
        req
    }

    async fn post_notification(&self, method: &str, session: &Session) -> Result<()> {
        let body = json!({ "jsonrpc": "2.0", "method": method });
        let resp = self
            .post(&body, session)
            .send()
            .await
            .map_err(|e| self.transport_error(e))?;
        if !resp.status().is_success() {
            bail!(
                "MCP server {}: {} returned HTTP {}",
                self.name,
                method,
                resp.status()
            );
        }
        Ok(())
    }

    /// Posts one request on `session` and returns its result and the
    /// session id header.
    async fn post_request(
        &self,
        method: &str,
        params: Value,
        session: &Session,
    ) -> Result<(Value, Option<String>)> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let body = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        let mut resp = self
            .post(&body, session)
            .send()
            .await
            .map_err(|e| self.transport_error(e))?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND && session.id.is_some() {
            return Err(SessionExpired.into());
        }
        if !status.is_success() {
            bail!(
                "MCP server {}: {} returned HTTP {}",
                self.name,
                method,
                status
            );
        }
        let session_id = resp
            .headers()
            .get(SESSION_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let is_sse = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.starts_with("text/event-stream"));

        let message = if is_sse {
            self.read_sse_response(&mut resp, id).await?
        } else {
            let bytes = self.read_capped(&mut resp).await?;
            serde_json::from_slice::<Value>(&bytes)
                .map_err(|e| anyhow!("MCP server {}: invalid JSON response: {}", self.name, e))?
        };
        Ok((self.unwrap_response(message, id)?, session_id))
    }

    /// Reads a body up to [`MAX_RESPONSE_BYTES`].
    async fn read_capped(&self, resp: &mut reqwest::Response) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        while let Some(chunk) = resp.chunk().await.map_err(|e| self.transport_error(e))? {
            buf.extend_from_slice(&chunk);
            self.check_size(buf.len())?;
        }
        Ok(buf)
    }

    fn check_size(&self, len: usize) -> Result<()> {
        if len > MAX_RESPONSE_BYTES {
            bail!(
                "MCP server {}: response exceeds {} bytes",
                self.name,
                MAX_RESPONSE_BYTES
            );
        }
        Ok(())
    }

    /// Reads an SSE stream until the event answering request `id` arrives.
    /// Server notifications and requests on the stream are skipped; an event
    /// that isn't valid JSON fails the request rather than being skipped.
    async fn read_sse_response(&self, resp: &mut reqwest::Response, id: u64) -> Result<Value> {
        let mut events = SseEvents::default();
        let mut total = 0;
        while let Some(chunk) = resp.chunk().await.map_err(|e| self.transport_error(e))? {
            total += chunk.len();
            self.check_size(total)?;
            for message in events.feed(&chunk) {
                let message = message.map_err(|e| {
                    anyhow!(
                        "MCP server {}: invalid JSON in event stream: {}",
                        self.name,
                        e
                    )
                })?;
                if message["id"] == json!(id) {
                    return Ok(message);
                }
            }
        }
        bail!(
            "MCP server {}: event stream ended without a response",
            self.name
        )
    }

    fn unwrap_response(&self, message: Value, id: u64) -> Result<Value> {
        if message["id"] != json!(id) {
            bail!(
                "MCP server {}: response id does not match request",
                self.name
            );
        }
        if let Some(err) = message.get("error") {
            let text = err["message"].as_str().unwrap_or("unknown error");
            bail!("MCP server {}: {}", self.name, self.redact(text));
        }
        message
            .get("result")
            .cloned()
            .ok_or_else(|| anyhow!("MCP server {}: response has no result", self.name))
    }

    /// Redacts server text before it reaches a log or a model: the token
    /// itself, wherever the server echoed it, and anything else that looks
    /// like a credential.
    fn redact(&self, text: &str) -> String {
        let text = match &self.token {
            Some(token) if !token.is_empty() => text.replace(token.as_str(), "[REDACTED]"),
            _ => text.to_string(),
        };
        redact_secret(&text)
    }

    fn transport_error(&self, e: reqwest::Error) -> anyhow::Error {
        // reqwest errors name the URL. Tokens travel in a header, but a URL
        // can carry credentials too, so redact before the text reaches a log
        // or a model.
        anyhow!(
            "MCP server {}: {}",
            self.name,
            redact_secret(&e.without_url().to_string())
        )
    }

    /// Lists the server's tools, following pagination.
    async fn list_tools(&self) -> Result<Vec<Value>> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_LIST_PAGES {
            let params = match &cursor {
                Some(c) => json!({ "cursor": c }),
                None => json!({}),
            };
            let result = self.request("tools/list", params).await?;
            if let Some(page) = result["tools"].as_array() {
                tools.extend(page.iter().cloned());
            }
            cursor = result["nextCursor"].as_str().map(str::to_string);
            if cursor.is_none() {
                return Ok(tools);
            }
        }
        warn!(
            "MCP server {}: stopped listing tools after {} pages",
            self.name, MAX_LIST_PAGES
        );
        Ok(tools)
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value> {
        self.request(
            "tools/call",
            json!({ "name": name, "arguments": arguments }),
        )
        .await
    }
}

/// The server dropped the session (HTTP 404 on a request carrying its id).
#[derive(Debug)]
struct SessionExpired;

impl std::fmt::Display for SessionExpired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MCP session expired")
    }
}

impl std::error::Error for SessionExpired {}

/// An incremental SSE parser: [`SseEvents::feed`] takes the next chunk and
/// returns the JSON messages of the events it completes. Each byte is
/// scanned once, however the stream is chunked.
#[derive(Default)]
struct SseEvents {
    /// Bytes of the event still being received, with CRs removed.
    pending: Vec<u8>,
    /// How far `pending` has been searched for an event boundary.
    scanned: usize,
}

impl SseEvents {
    fn feed(&mut self, chunk: &[u8]) -> Vec<Result<Value, serde_json::Error>> {
        self.pending.extend(chunk.iter().filter(|&&b| b != b'\r'));
        let mut messages = Vec::new();
        loop {
            let from = self.scanned.saturating_sub(1);
            let Some(end) = self.pending[from..]
                .windows(2)
                .position(|w| w == b"\n\n")
                .map(|p| from + p)
            else {
                self.scanned = self.pending.len();
                return messages;
            };
            let event: Vec<u8> = self.pending.drain(..end + 2).collect();
            self.scanned = 0;
            let event = String::from_utf8_lossy(&event[..end]);
            let data: Vec<&str> = event
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|d| d.strip_prefix(' ').unwrap_or(d))
                .collect();
            if !data.is_empty() {
                messages.push(serde_json::from_str(&data.join("\n")));
            }
        }
    }
}

/// One server tool, exposed to the model as `mcp_<server>_<tool>`.
#[derive(Clone)]
pub struct McpTool {
    client: Arc<McpClient>,
    remote_name: String,
    name: String,
    description: String,
    parameters: Value,
    max_output_bytes: usize,
    /// Review stages that may see and call this tool.
    stages: Vec<String>,
}

// Hand-written so the client, which holds the token, is never printed.
impl std::fmt::Debug for McpTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpTool")
            .field("name", &self.name)
            .field("stages", &self.stages)
            .finish_non_exhaustive()
    }
}

impl McpTool {
    /// Builds the tool for one `tools/list` entry, or explains why it is
    /// skipped.
    fn from_listing(
        client: &Arc<McpClient>,
        settings: &McpServerSettings,
        listing: &Value,
    ) -> Result<Self, String> {
        let remote_name = listing["name"].as_str().unwrap_or_default();
        let valid = !remote_name.is_empty()
            && remote_name.len() <= MAX_TOOL_NAME_BYTES
            && remote_name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
        if !valid {
            return Err(format!("invalid tool name {:?}", remote_name));
        }
        // ToolBox::call lowercases names, so the exposed name is lowercase.
        let name = format!(
            "mcp_{}_{}",
            settings.name,
            remote_name.to_ascii_lowercase().replace('-', "_")
        );
        if name.len() > MAX_TOOL_NAME_BYTES {
            return Err(format!("tool name {:?} is too long", name));
        }
        let remote_description = listing["description"].as_str().unwrap_or_default();
        let description = format!(
            "[MCP server {}] {}",
            settings.name,
            truncate_utf8(remote_description, MAX_DESCRIPTION_BYTES)
        );
        let parameters = match listing.get("inputSchema") {
            Some(schema) if schema.is_object() => {
                let mut schema = schema.clone();
                if let Some(obj) = schema.as_object_mut() {
                    obj.remove("$schema");
                }
                schema
            }
            _ => json!({ "type": "object", "properties": {} }),
        };
        Ok(Self {
            client: client.clone(),
            remote_name: remote_name.to_string(),
            name,
            description,
            parameters,
            max_output_bytes: settings.max_output_bytes,
            stages: settings.stages.clone(),
        })
    }

    /// Converts a `tools/call` result into the tool's JSON output. Server
    /// text is redacted (the token itself, and anything that looks like a
    /// credential) and capped at `max_output_bytes`, with a visible marker
    /// where it was cut.
    fn format_result(&self, result: &Value) -> Value {
        let structured = result.get("structuredContent");
        let parts: Vec<String> = match result.get("content").and_then(Value::as_array) {
            Some(items) => items
                .iter()
                .map(
                    |item| match (item["type"].as_str(), item["text"].as_str()) {
                        (Some("text"), Some(text)) => text.to_string(),
                        (Some("text"), None) => "[malformed text content]".to_string(),
                        (Some(other), _) => format!("[{} content omitted]", other),
                        (None, _) => "[malformed content item]".to_string(),
                    },
                )
                .collect(),
            None if structured.is_some() => Vec::new(),
            None => {
                return json!({
                    "error": "The MCP server returned a malformed result (no content).",
                    "source": self.source(),
                });
            }
        };
        let text = if parts.is_empty() {
            structured.map(Value::to_string).unwrap_or_default()
        } else {
            parts.join("\n\n")
        };
        let text = self.client.redact(&text);
        let (text, truncated) = self.cap(&text);
        let key = if result["isError"].as_bool() == Some(true) {
            "error"
        } else {
            "content"
        };
        let mut out = json!({ "source": self.source(), "truncated": truncated });
        out[key] = json!(text);
        if truncated {
            out["next_page_hint"] =
                json!("The result was truncated. Narrow the query or request a smaller part.");
        }
        out
    }

    /// Caps text at `max_output_bytes`, appending a marker when it is cut.
    fn cap(&self, text: &str) -> (String, bool) {
        if text.len() <= self.max_output_bytes {
            return (text.to_string(), false);
        }
        let kept = truncate_utf8(text, self.max_output_bytes);
        (
            format!(
                "{kept}\n... [Output truncated. Displaying first {} of {} bytes] ...\n",
                kept.len(),
                text.len()
            ),
            true,
        )
    }

    /// Review stages that may see and call this tool.
    pub fn stages(&self) -> &[String] {
        &self.stages
    }

    fn source(&self) -> String {
        format!("mcp:{}", self.client.name)
    }
}

#[async_trait]
impl LlmTool<SashikoToolContext> for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    async fn call(&self, args: Value, _context: &SashikoToolContext) -> Result<Value> {
        let arguments = if args.is_object() { args } else { json!({}) };
        let result = self.client.call_tool(&self.remote_name, arguments).await?;
        Ok(self.format_result(&result))
    }
}

/// A configured server's hint for the stages that see its tools.
#[derive(Clone, Debug)]
pub struct McpPromptHint {
    pub server: String,
    pub hint: Option<String>,
    pub tools: Vec<String>,
    pub stages: Vec<String>,
}

/// Every tool discovered from the configured servers.
#[derive(Clone, Default)]
pub struct McpTools {
    tools: Vec<McpTool>,
    hints: Vec<McpPromptHint>,
}

impl McpTools {
    /// Connects to every configured server concurrently and lists its
    /// allowed tools. A server that fails is logged and skipped, so a review
    /// never fails because a reference server is down.
    pub async fn discover(settings: &McpSettings) -> Self {
        let found = futures::future::join_all(settings.servers.iter().map(discover_server)).await;
        let mut all = Self::default();
        for (server, result) in settings.servers.iter().zip(found) {
            match result {
                Ok(tools) if tools.is_empty() => {
                    warn!(
                        "MCP server {}: none of the allowed tools are offered",
                        server.name
                    );
                }
                Ok(tools) => {
                    info!(
                        "MCP server {}: {} tool(s) available",
                        server.name,
                        tools.len()
                    );
                    all.hints.push(McpPromptHint {
                        server: server.name.clone(),
                        hint: server.prompt_hint.clone(),
                        tools: tools.iter().map(|t| t.name.clone()).collect(),
                        stages: server.stages.clone(),
                    });
                    all.tools.extend(tools);
                }
                Err(e) => warn!("MCP server {} skipped: {}", server.name, e),
            }
        }
        all
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn tools(&self) -> &[McpTool] {
        &self.tools
    }

    pub fn hints(&self) -> &[McpPromptHint] {
        &self.hints
    }
}

#[cfg(test)]
impl McpTools {
    /// Tools for a server that is never contacted, for tests of how tools
    /// are exposed rather than how they are called.
    pub(crate) fn offline(settings: &McpServerSettings, remote_names: &[&str]) -> Self {
        let client = Arc::new(McpClient::new(settings, None).expect("a client builds offline"));
        let tools: Vec<McpTool> = remote_names
            .iter()
            .map(|name| {
                McpTool::from_listing(&client, settings, &json!({ "name": name }))
                    .expect("test tool names are valid")
            })
            .collect();
        let hints = vec![McpPromptHint {
            server: settings.name.clone(),
            hint: settings.prompt_hint.clone(),
            tools: tools.iter().map(|t| t.name.clone()).collect(),
            stages: settings.stages.clone(),
        }];
        Self { tools, hints }
    }
}

/// Reads the bearer token from the variable the settings name, if any.
fn resolve_token(settings: &McpServerSettings) -> Result<Option<String>> {
    let Some(var) = &settings.bearer_token_env else {
        return Ok(None);
    };
    match std::env::var(var) {
        Ok(token) if !token.is_empty() => Ok(Some(token)),
        _ => Err(anyhow!(
            "environment variable {} named by bearer_token_env is not set",
            var
        )),
    }
}

async fn discover_server(settings: &McpServerSettings) -> Result<Vec<McpTool>> {
    discover_server_with_token(settings, resolve_token(settings)?).await
}

async fn discover_server_with_token(
    settings: &McpServerSettings,
    token: Option<String>,
) -> Result<Vec<McpTool>> {
    let client = Arc::new(McpClient::new(settings, token)?);
    client.initialize().await?;
    let listings = client.list_tools().await?;
    let mut tools: Vec<McpTool> = Vec::new();
    for listing in &listings {
        let remote = listing["name"].as_str().unwrap_or_default();
        if !settings.allowed_tools.iter().any(|t| t == remote) {
            continue;
        }
        match McpTool::from_listing(&client, settings, listing) {
            Ok(tool) if tools.iter().any(|t| t.name == tool.name) => {
                warn!("MCP server {}: duplicate tool {}", settings.name, tool.name);
            }
            Ok(tool) => tools.push(tool),
            Err(reason) => warn!("MCP server {}: skipping tool: {}", settings.name, reason),
        }
    }
    for wanted in &settings.allowed_tools {
        if !tools.iter().any(|t| &t.remote_name == wanted) {
            warn!(
                "MCP server {}: allowed tool {} is not offered",
                settings.name, wanted
            );
        }
    }
    tools.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(tools)
}

/// Truncates to at most `max` bytes on a character boundary.
fn truncate_utf8(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(all(test, feature = "server"))]
mod tests;
