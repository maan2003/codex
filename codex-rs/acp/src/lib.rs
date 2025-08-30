use acp::AgentSideConnection;
use acp::SessionUpdate;
use agent_client_protocol as acp;
use agent_client_protocol::CancelNotification;
use agent_client_protocol::Client;
use agent_client_protocol::ContentBlock;
use agent_client_protocol::PermissionOption;
use agent_client_protocol::PermissionOptionId;
use agent_client_protocol::PermissionOptionKind;
use agent_client_protocol::Plan;
use agent_client_protocol::PlanEntry;
use agent_client_protocol::PlanEntryPriority;
use agent_client_protocol::PlanEntryStatus;
use agent_client_protocol::RequestPermissionOutcome;
use agent_client_protocol::RequestPermissionRequest;
use agent_client_protocol::SessionId;
use agent_client_protocol::SessionNotification;
use agent_client_protocol::StopReason;
use agent_client_protocol::ToolCall;
use agent_client_protocol::ToolCallContent;
use agent_client_protocol::ToolCallId;
use agent_client_protocol::ToolCallLocation;
use agent_client_protocol::ToolCallStatus;
use agent_client_protocol::ToolCallUpdate;
use agent_client_protocol::ToolCallUpdateFields;
use agent_client_protocol::ToolKind;
use anyhow::Result;
use codex_common::CliConfigOverrides;
use codex_core::AuthManager;
use codex_core::CodexConversation;
use codex_core::ConversationManager;
use codex_core::config::Config;
use codex_core::config::ConfigOverrides;
use codex_core::config_types::McpServerConfig;
use codex_core::plan_tool::StepStatus;
use codex_core::protocol::AskForApproval;
use codex_core::protocol::EventMsg;
use codex_core::protocol::InputItem;
use codex_core::protocol::McpInvocation;
use codex_core::protocol::Op;
use codex_core::protocol::ReviewDecision;
use codex_protocol::config_types::SandboxMode;
use codex_protocol::mcp_protocol::ConversationId;
use codex_protocol::parse_command::ParsedCommand as Parsed;
use shlex::try_join;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::Weak;
use tokio::task::LocalSet;
use tokio_util::compat::TokioAsyncReadCompatExt;
use tokio_util::compat::TokioAsyncWriteCompatExt;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

fn escape_command(command: &[String]) -> String {
    try_join(command.iter().map(|s| s.as_str())).unwrap_or_else(|_| command.join(" "))
}

fn strip_bash_lc_and_escape(command: &[String]) -> String {
    match command {
        [first, second, third] if first == "bash" && second == "-lc" => third.clone(),
        _ => escape_command(command),
    }
}

trait AcpResultExt<T, E> {
    fn acp_internal_err(self) -> Result<T, acp::Error>;
}

impl<T, E> AcpResultExt<T, E> for std::result::Result<T, E>
where
    E: std::error::Error,
{
    #[inline]
    fn acp_internal_err(self) -> Result<T, acp::Error> {
        self.map_err(acp::Error::into_internal_error)
    }
}

struct CodexAgent {
    conversation_manager: ConversationManager,
    tool_calls: RefCell<HashMap<ToolCallId, ToolCall>>,
    client: Weak<AgentSideConnection>,
    codex_linux_sandbox_exe: Option<PathBuf>,
    cli_kv_overrides: Vec<(String, toml::Value)>,
}

impl CodexAgent {
    fn new(
        client: Weak<AgentSideConnection>,
        codex_linux_sandbox_exe: Option<PathBuf>,
        cli_kv_overrides: Vec<(String, toml::Value)>,
        conversation_manager: ConversationManager,
    ) -> Self {
        Self {
            conversation_manager,
            tool_calls: RefCell::new(HashMap::new()),
            client,
            codex_linux_sandbox_exe,
            cli_kv_overrides,
        }
    }

    fn get_client(&self) -> Result<Arc<AgentSideConnection>, acp::Error> {
        self.client.upgrade().ok_or_else(acp::Error::internal_error)
    }

    async fn send_session_notification(
        &self,
        session_id: SessionId,
        update: SessionUpdate,
    ) -> Result<(), acp::Error> {
        let client = self.get_client()?;
        client
            .session_notification(SessionNotification { session_id, update })
            .await
    }
}

fn build_tool_call_from_exec_begin(msg: &codex_core::protocol::ExecCommandBeginEvent) -> ToolCall {
    let mut tc = ToolCall {
        id: ToolCallId(msg.call_id.clone().into()),
        title: strip_bash_lc_and_escape(&msg.command),
        kind: ToolKind::Execute,
        status: ToolCallStatus::InProgress,
        content: vec![],
        locations: vec![],
        raw_input: None,
        raw_output: None,
    };

    match msg.parsed_cmd.as_slice() {
        [Parsed::Read { name, .. }] => {
            tc.title = format!("Read {name}");
            tc.kind = ToolKind::Read;
            if !name.is_empty() {
                tc.locations.push(ToolCallLocation {
                    path: name.into(),
                    line: None,
                });
            }
        }
        [Parsed::ListFiles { cmd, path }] => {
            tc.title = match path {
                Some(p) => format!("List {p}"),
                None => format!("List {cmd}"),
            };
            tc.kind = ToolKind::Read;
        }
        [Parsed::Search { query, path, cmd }] => {
            tc.title = match (query, path) {
                (Some(q), Some(p)) => format!("Search {q} in {p}"),
                (Some(q), None) => format!("Search {q}"),
                (None, Some(p)) => format!("Search {p}"),
                (None, None) => format!("Search {cmd}"),
            };
            tc.kind = ToolKind::Search;
        }
        _ => {}
    }

    tc
}

impl acp::Agent for CodexAgent {
    async fn initialize(
        &self,
        _arguments: acp::InitializeRequest,
    ) -> Result<acp::InitializeResponse, acp::Error> {
        Ok(acp::InitializeResponse {
            protocol_version: acp::V1,
            agent_capabilities: acp::AgentCapabilities::default(),
            auth_methods: Vec::new(),
        })
    }

    async fn authenticate(&self, _arguments: acp::AuthenticateRequest) -> Result<(), acp::Error> {
        Ok(())
    }

    async fn new_session(
        &self,
        arguments: acp::NewSessionRequest,
    ) -> Result<acp::NewSessionResponse, acp::Error> {
        let overrides = ConfigOverrides {
            include_plan_tool: Some(true),
            approval_policy: Some(AskForApproval::OnRequest),
            sandbox_mode: Some(SandboxMode::WorkspaceWrite),
            cwd: Some(arguments.cwd.clone()),
            codex_linux_sandbox_exe: self.codex_linux_sandbox_exe.clone(),
            ..ConfigOverrides::default()
        };
        let mut config = Config::load_with_cli_overrides(self.cli_kv_overrides.clone(), overrides)
            .acp_internal_err()?;

        for server in arguments.mcp_servers {
            let server_cfg = McpServerConfig {
                command: server.command.to_string_lossy().to_string(),
                args: server.args,
                env: Some(server.env.into_iter().map(|v| (v.name, v.value)).collect()),
                startup_timeout_ms: None,
            };
            config.mcp_servers.insert(server.name, server_cfg);
        }

        let new_conv = self
            .conversation_manager
            .new_conversation(config)
            .await
            .acp_internal_err()?;

        let session_id = SessionId(new_conv.conversation_id.to_string().into());
        Ok(acp::NewSessionResponse { session_id })
    }

    async fn load_session(&self, _arguments: acp::LoadSessionRequest) -> Result<(), acp::Error> {
        Err(acp::Error::method_not_found())
    }

    async fn prompt(
        &self,
        arguments: acp::PromptRequest,
    ) -> Result<acp::PromptResponse, acp::Error> {
        let conv_id = Uuid::parse_str(arguments.session_id.0.as_ref())
            .map(ConversationId::from)
            .map_err(|_| acp::Error::invalid_params().with_data("Invalid session id"))?;

        let conversation: Arc<CodexConversation> = self
            .conversation_manager
            .get_conversation(conv_id)
            .await
            .acp_internal_err()?;

        let items = arguments
            .prompt
            .into_iter()
            .map(|c| match c {
                ContentBlock::Text(t) => Ok(InputItem::Text { text: t.text }),
                _ => Err(acp::Error::invalid_params().with_data("Unsupported content block type")),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let op = Op::UserInput { items };
        let submission_id = conversation.submit(op).await.acp_internal_err()?;

        let stop_reason = loop {
            let event = conversation.next_event().await.acp_internal_err()?;
            if event.id != submission_id {
                continue;
            }

            match event.msg {
                EventMsg::AgentMessageDelta(msg) => {
                    self.send_session_notification(
                        arguments.session_id.clone(),
                        SessionUpdate::AgentMessageChunk {
                            content: msg.delta.into(),
                        },
                    )
                    .await?;
                }
                EventMsg::AgentReasoningDelta(msg) => {
                    self.send_session_notification(
                        arguments.session_id.clone(),
                        SessionUpdate::AgentThoughtChunk {
                            content: msg.delta.into(),
                        },
                    )
                    .await?;
                }
                EventMsg::ExecCommandBegin(msg) => {
                    let tool_call_id = ToolCallId(msg.call_id.clone().into());
                    let tool_call = build_tool_call_from_exec_begin(&msg);
                    self.tool_calls
                        .borrow_mut()
                        .insert(tool_call_id, tool_call.clone());

                    self.send_session_notification(
                        arguments.session_id.clone(),
                        SessionUpdate::ToolCall(tool_call),
                    )
                    .await?;
                }
                EventMsg::ExecCommandOutputDelta(msg) => {
                    let tool_call_id = ToolCallId(msg.call_id.clone().into());
                    if let Some(tool_call) = self.tool_calls.borrow_mut().get_mut(&tool_call_id) {
                        let text = String::from_utf8_lossy(&msg.chunk).to_string();
                        tool_call.content.push(ToolCallContent::Content {
                            content: text.into(),
                        });

                        // Send an update with the full accumulated content
                        self.send_session_notification(
                            arguments.session_id.clone(),
                            SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                                id: tool_call_id.clone(),
                                fields: ToolCallUpdateFields {
                                    content: Some(tool_call.content.clone()),
                                    ..Default::default()
                                },
                            }),
                        )
                        .await?;
                    }
                }
                EventMsg::ExecCommandEnd(msg) => {
                    let tool_call_id = ToolCallId(msg.call_id.clone().into());
                    if let Some(mut tool_call) = self.tool_calls.borrow_mut().remove(&tool_call_id)
                    {
                        tool_call.status = if msg.exit_code == 0 {
                            ToolCallStatus::Completed
                        } else {
                            ToolCallStatus::Failed
                        };

                        self.send_session_notification(
                            arguments.session_id.clone(),
                            SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                                id: tool_call_id.clone(),
                                fields: ToolCallUpdateFields {
                                    status: Some(tool_call.status.clone()),
                                    ..Default::default()
                                },
                            }),
                        )
                        .await?;
                    }
                }
                EventMsg::McpToolCallBegin(msg) => {
                    let tool_call_id = ToolCallId(msg.call_id.clone().into());
                    let McpInvocation { server, tool, .. } = msg.invocation;
                    let tool_call = ToolCall {
                        id: tool_call_id.clone(),
                        title: format!("{}::{}", server, tool),
                        kind: ToolKind::Fetch,
                        status: ToolCallStatus::InProgress,
                        content: vec![],
                        locations: vec![],
                        raw_input: None,
                        raw_output: None,
                    };
                    self.tool_calls
                        .borrow_mut()
                        .insert(tool_call_id, tool_call.clone());

                    self.send_session_notification(
                        arguments.session_id.clone(),
                        SessionUpdate::ToolCall(tool_call),
                    )
                    .await?;
                }
                EventMsg::McpToolCallEnd(msg) => {
                    let tool_call_id = ToolCallId(msg.call_id.clone().into());
                    if let Some(mut tool_call) = self.tool_calls.borrow_mut().remove(&tool_call_id)
                    {
                        tool_call.status = if msg.is_success() {
                            ToolCallStatus::Completed
                        } else {
                            ToolCallStatus::Failed
                        };

                        self.send_session_notification(
                            arguments.session_id.clone(),
                            SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                                id: tool_call_id.clone(),
                                fields: ToolCallUpdateFields {
                                    status: Some(tool_call.status.clone()),
                                    ..Default::default()
                                },
                            }),
                        )
                        .await?;
                    }
                }
                EventMsg::WebSearchBegin(msg) => {
                    let tool_call_id = ToolCallId(msg.call_id.into());
                    let tool_call = ToolCall {
                        id: tool_call_id.clone(),
                        title: "Web Search".to_string(),
                        kind: ToolKind::Search,
                        status: ToolCallStatus::InProgress,
                        content: vec![],
                        locations: vec![],
                        raw_input: None,
                        raw_output: None,
                    };
                    self.tool_calls
                        .borrow_mut()
                        .insert(tool_call_id, tool_call.clone());

                    self.send_session_notification(
                        arguments.session_id.clone(),
                        SessionUpdate::ToolCall(tool_call),
                    )
                    .await?;
                }
                EventMsg::WebSearchEnd(msg) => {
                    let tool_call_id = ToolCallId(msg.call_id.into());
                    if let Some(mut tool_call) = self.tool_calls.borrow_mut().remove(&tool_call_id)
                    {
                        tool_call.status = ToolCallStatus::Completed;
                        tool_call.content.push(ToolCallContent::Content {
                            content: msg.query.into(),
                        });

                        self.send_session_notification(
                            arguments.session_id.clone(),
                            SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                                id: tool_call_id.clone(),
                                fields: ToolCallUpdateFields {
                                    status: Some(ToolCallStatus::Completed),
                                    content: Some(tool_call.content.clone()),
                                    ..Default::default()
                                },
                            }),
                        )
                        .await?;
                    }
                }
                EventMsg::PlanUpdate(msg) => {
                    let plan = Plan {
                        entries: msg
                            .plan
                            .into_iter()
                            .map(|p| PlanEntry {
                                content: p.step,
                                priority: PlanEntryPriority::Medium,
                                status: match p.status {
                                    StepStatus::Pending => PlanEntryStatus::Pending,
                                    StepStatus::InProgress => PlanEntryStatus::InProgress,
                                    StepStatus::Completed => PlanEntryStatus::Completed,
                                },
                            })
                            .collect(),
                    };

                    self.send_session_notification(
                        arguments.session_id.clone(),
                        SessionUpdate::Plan(plan),
                    )
                    .await?;
                }
                EventMsg::PatchApplyBegin(msg) => {
                    let tool_call_id = ToolCallId(msg.call_id.into());
                    let mut tool_call = ToolCall {
                        id: tool_call_id.clone(),
                        title: "Apply Patch".to_string(),
                        kind: ToolKind::Edit,
                        status: ToolCallStatus::InProgress,
                        content: vec![],
                        locations: vec![],
                        raw_input: None,
                        raw_output: None,
                    };
                    // Attach raw_input: patch changes, best effort
                    if let Ok(val) = serde_json::to_value(&msg.changes) {
                        tool_call.raw_input = Some(val);
                    }
                    self.tool_calls
                        .borrow_mut()
                        .insert(tool_call_id.clone(), tool_call.clone());

                    self.send_session_notification(
                        arguments.session_id.clone(),
                        SessionUpdate::ToolCall(tool_call),
                    )
                    .await?;
                }
                EventMsg::PatchApplyEnd(msg) => {
                    let tool_call_id = ToolCallId(msg.call_id.clone().into());
                    if self.tool_calls.borrow().contains_key(&tool_call_id) {
                        let status = if msg.success {
                            ToolCallStatus::Completed
                        } else {
                            ToolCallStatus::Failed
                        };
                        self.send_session_notification(
                            arguments.session_id.clone(),
                            SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                                id: tool_call_id.clone(),
                                fields: ToolCallUpdateFields {
                                    status: Some(status),
                                    ..Default::default()
                                },
                            }),
                        )
                        .await?;
                        // Mark finished in our local map; keep it around if needed for content history.
                        if let Some(tc) = self.tool_calls.borrow_mut().get_mut(&tool_call_id) {
                            tc.status = status;
                        }
                    }
                }
                EventMsg::TurnDiff(_msg) => {
                    // Intentionally ignored for now.
                }
                EventMsg::ExecApprovalRequest(msg) => {
                    let client = self.get_client()?;
                    let response = client
                        .request_permission(RequestPermissionRequest {
                            session_id: arguments.session_id.clone(),
                            tool_call: ToolCallUpdate {
                                id: ToolCallId(msg.call_id.clone().into()),
                                fields: ToolCallUpdateFields {
                                    title: Some(strip_bash_lc_and_escape(&msg.command)),
                                    ..Default::default()
                                },
                            },
                            options: vec![
                                PermissionOption {
                                    id: PermissionOptionId("allow_once".into()),
                                    name: "Allow Once".to_string(),
                                    kind: PermissionOptionKind::AllowOnce,
                                },
                                PermissionOption {
                                    id: PermissionOptionId("reject_once".into()),
                                    name: "Reject Once".to_string(),
                                    kind: PermissionOptionKind::RejectOnce,
                                },
                            ],
                        })
                        .await?;

                    let decision = match response.outcome {
                        RequestPermissionOutcome::Selected { option_id } => {
                            if option_id.0.as_ref() == "allow_once" {
                                ReviewDecision::Approved
                            } else {
                                ReviewDecision::Denied
                            }
                        }
                        RequestPermissionOutcome::Cancelled => ReviewDecision::Denied,
                    };

                    conversation
                        .submit(Op::ExecApproval {
                            id: event.id,
                            decision,
                        })
                        .await
                        .acp_internal_err()?;
                }
                EventMsg::ApplyPatchApprovalRequest(msg) => {
                    let client = self.get_client()?;
                    let response = client
                        .request_permission(RequestPermissionRequest {
                            session_id: arguments.session_id.clone(),
                            tool_call: ToolCallUpdate {
                                id: ToolCallId(msg.call_id.clone().into()),
                                fields: ToolCallUpdateFields {
                                    title: Some("Apply patch".to_string()),
                                    ..Default::default()
                                },
                            },
                            options: vec![
                                PermissionOption {
                                    id: PermissionOptionId("allow_once".into()),
                                    name: "Allow Once".to_string(),
                                    kind: PermissionOptionKind::AllowOnce,
                                },
                                PermissionOption {
                                    id: PermissionOptionId("reject_once".into()),
                                    name: "Reject Once".to_string(),
                                    kind: PermissionOptionKind::RejectOnce,
                                },
                            ],
                        })
                        .await?;

                    let decision = match response.outcome {
                        RequestPermissionOutcome::Selected { option_id } => {
                            if option_id.0.as_ref() == "allow_once" {
                                ReviewDecision::Approved
                            } else {
                                ReviewDecision::Denied
                            }
                        }
                        RequestPermissionOutcome::Cancelled => ReviewDecision::Denied,
                    };

                    conversation
                        .submit(Op::PatchApproval {
                            id: msg.call_id,
                            decision,
                        })
                        .await
                        .acp_internal_err()?;
                }
                EventMsg::TaskComplete(_) => {
                    break StopReason::EndTurn;
                }
                EventMsg::TurnAborted(_) => {
                    break StopReason::Cancelled;
                }
                _ => {}
            }
        };

        Ok(acp::PromptResponse { stop_reason })
    }

    async fn cancel(&self, args: CancelNotification) -> Result<(), acp::Error> {
        let conv_id = Uuid::parse_str(args.session_id.0.as_ref())
            .map(ConversationId::from)
            .map_err(|_| acp::Error::invalid_params().with_data("Invalid session id"))?;
        if let Ok(conversation) = self.conversation_manager.get_conversation(conv_id).await {
            conversation
                .submit(Op::Interrupt)
                .await
                .acp_internal_err()?;
        }
        Ok(())
    }
}

pub async fn run_main(
    codex_linux_sandbox_exe: Option<PathBuf>,
    cli_config_overrides: CliConfigOverrides,
) -> Result<()> {
    // Install a simple subscriber so `tracing` output is visible.  Users can
    // control the log level with `RUST_LOG`.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(EnvFilter::from_str("debug")?)
        .init();

    let outgoing = tokio::io::stdout().compat_write();
    let incoming = tokio::io::stdin().compat();

    let local_set = LocalSet::new();
    local_set
        .run_until(async move {
            // Build the connection with a Weak reference passed into the agent
            let mut handle_io_opt = None;
            let handle_io_opt_ref = &mut handle_io_opt;

            let cli_kv_overrides = cli_config_overrides
                .parse_overrides()
                .map_err(|e| anyhow::anyhow!("error parsing -c overrides: {e}"))?;

            let codex_home = codex_core::config::find_codex_home()?;
            let auth_manager = AuthManager::shared(codex_home);
            let conversation_manager = ConversationManager::new(auth_manager);
            let _conn_arc: Arc<AgentSideConnection> = Arc::new_cyclic(move |weak| {
                let agent = CodexAgent::new(
                    weak.clone(),
                    codex_linux_sandbox_exe,
                    cli_kv_overrides,
                    conversation_manager,
                );
                let (conn, handle_io) =
                    AgentSideConnection::new(agent, outgoing, incoming, |fut| {
                        tokio::task::spawn_local(fut);
                    });

                *handle_io_opt_ref = Some(handle_io);
                conn
            });

            // Await I/O task completion
            handle_io_opt.expect("io future set").await
        })
        .await?;
    Ok(())
}
