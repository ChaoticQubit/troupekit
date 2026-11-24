use crate::budget::{self, Limits};
use crate::dsl::Call;
use crate::error::AgentError;
use crate::eval::{judge_request, judge_verdict, repair_message};
use crate::io::{AgentIo, ModelHandle};
use crate::persona::Persona;
use crate::schema::{parse_json, validate};
use futures_util::future::join_all;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use troupekit_core::event::{Event, EventBody, PauseReason, Usage};
use troupekit_core::ids::RunId;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_core::store::{Checkpoint, CheckpointStore};
use troupekit_providers::{CompletionRequest, CompletionResponse, Delta, StopReason};
use troupekit_tools::{ToolContext, ToolOutput};

/// Version of the serialized run.
/// `from_checkpoint` refuses any other, so a deploy never resumes
/// a run it cannot read.
pub const RUN_FORMAT: u32 = 1;

/// A tool call the model asked for and the run has not answered yet.
/// Kept in the run state, so a resumed run re-runs the whole batch:
/// tool calls are at least once across a crash.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PendingCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Where a run stands. `CallModel`, `CallTools`, `Check` and `Repair` each
/// do one step of work; `Done` and `Failed` are final; `Paused` waits for
/// `resume` and then returns to the state in `resume`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum RunState {
    CallModel,
    CallTools {
        calls: Vec<PendingCall>,
    },
    Check {
        answer: String,
    },
    Repair {
        errors: Vec<String>,
    },
    Done {
        output: Value,
    },
    Failed {
        error: String,
    },
    Paused {
        reason: PauseReason,
        resume: Box<RunState>,
    },
}

impl RunState {
    fn step_name(&self) -> &'static str {
        match self {
            RunState::CallModel => "call_model",
            RunState::CallTools { .. } => "call_tools",
            RunState::Check { .. } => "check",
            RunState::Repair { .. } => "repair",
            RunState::Done { .. } | RunState::Failed { .. } | RunState::Paused { .. } => {
                unreachable!("only working states take steps")
            }
        }
    }
}

/// What one `step` did. `Continue`: step again. `Retry`: a transient
/// provider or ledger failure left the state unchanged; step again later.
/// `Paused`, `Done` and `Failed` repeat on every further step.
#[derive(Debug, Clone, PartialEq)]
pub enum StepOutcome {
    Continue,
    Retry {
        error: String,
        retry_after_ms: Option<u64>,
    },
    Paused(PauseReason),
    Done(Value),
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunSpec {
    persona: String,
    model: String,
    system: String,
    tools: Vec<String>,
    evals: Vec<Call>,
    output: Option<Value>,
    max_turns: u32,
    max_tokens: u32,
}

/// One agent run as plain serializable data: the persona snapshot, the
/// conversation, the state, counters, usage and the event sequence. The
/// host calls `step` and saves `checkpoint()` after every step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentRun {
    format: u32,
    run_id: RunId,
    spec: RunSpec,
    limits: Limits,
    messages: Vec<Message>,
    state: RunState,
    model_calls: u32,
    repaired: bool,
    usage: Usage,
    seq: u64,
    steps: u64,
}

impl AgentRun {
    /// Starts a run of `persona` on `input` under `limits` tightened by the
    /// persona budget. Checks before any model call that every tool, eval and
    /// model it names is available in `io`; returns `Setup` naming the file.
    pub async fn new(
        persona: &Persona,
        input: Vec<Message>,
        limits: Limits,
        io: &dyn AgentIo,
    ) -> Result<AgentRun, AgentError> {
        let setup = |m: String| AgentError::Setup(format!("{}: {m}", persona.source));
        if input.is_empty() {
            return Err(setup("a run needs at least one input message".into()));
        }
        if persona.instructions.trim().is_empty() {
            return Err(setup("instructions are not loaded".into()));
        }
        if let Some(t) = persona.tools.iter().find(|t| io.tools().get(t).is_none()) {
            return Err(setup(format!("tool `{t}` is not registered")));
        }
        let mut models = vec![persona.model.clone()];
        for e in &persona.evals {
            if e.name == "judge" {
                models.extend(e.args.get(2).and_then(Value::as_str).map(str::to_string));
            } else if io.evals().get(&e.name).is_none() {
                return Err(setup(format!("eval `{}` is not registered", e.name)));
            }
        }
        for m in &models {
            io.model(m).await.map_err(|e| setup(e.to_string()))?;
        }
        Ok(AgentRun {
            format: RUN_FORMAT,
            run_id: RunId::new(),
            spec: RunSpec {
                persona: persona.name.clone(),
                model: persona.model.clone(),
                system: persona.system_prompt(),
                tools: persona.tools.clone(),
                evals: persona.evals.clone(),
                output: persona.output.clone(),
                max_turns: persona.max_turns,
                max_tokens: persona.max_tokens,
            },
            limits: limits.with_budget(&persona.budget),
            messages: input,
            state: RunState::CallModel,
            model_calls: 0,
            repaired: false,
            usage: Usage::default(),
            seq: 0,
            steps: 0,
        })
    }

    /// Restores a run from `cp`. Returns `Checkpoint` when the state does
    /// not parse, has unknown fields, has another `RUN_FORMAT`, or does not
    /// match the checkpoint's run id and step.
    pub fn from_checkpoint(cp: &Checkpoint) -> Result<AgentRun, AgentError> {
        let run: AgentRun = serde_json::from_value(cp.state.clone())
            .map_err(|e| AgentError::Checkpoint(e.to_string()))?;
        if run.format != RUN_FORMAT {
            return Err(AgentError::Checkpoint(format!(
                "run format {} is not {RUN_FORMAT}",
                run.format
            )));
        }
        if run.run_id != cp.run_id || run.steps != cp.step {
            return Err(AgentError::Checkpoint(
                "checkpoint header does not match its state".into(),
            ));
        }
        Ok(run)
    }

    /// Snapshots the run for a `CheckpointStore`: step 0 before the first
    /// step, then one more after each step, whatever its outcome.
    /// Returns the checkpoint.
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            run_id: self.run_id,
            step: self.steps,
            state: serde_json::to_value(self).expect("run state serializes"),
        }
    }

    /// Returns the run's id.
    /// Every event and checkpoint of the run carries it.
    /// It never changes.
    pub fn run_id(&self) -> RunId {
        self.run_id
    }

    /// Returns the current state.
    /// Hosts read it to show progress or the final output.
    /// It changes only in `step` and `resume`.
    pub fn state(&self) -> &RunState {
        &self.state
    }

    /// Returns the conversation so far: input, replies, tool results
    /// and repair requests, in order.
    /// The system prompt is not part of it.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Returns tokens and cost of every model call so far.
    /// Judge calls are included; costs use the handle's price.
    /// Compare it with the run budget.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// Returns how many agent model calls ran.
    /// This is the count `max_turns` caps; the repair call counts,
    /// judge calls do not.
    pub fn model_calls(&self) -> u32 {
        self.model_calls
    }

    /// Leaves a pause with new `limits` (for example after an owner raised
    /// a budget), back in the state that paused. The next step checks the
    /// budget again. Returns `Setup` when the run is not paused.
    pub fn resume(&mut self, limits: Limits) -> Result<(), AgentError> {
        let RunState::Paused { resume, .. } = &self.state else {
            return Err(AgentError::Setup("run is not paused".into()));
        };
        self.state = (**resume).clone();
        self.limits = limits;
        Ok(())
    }

    /// Runs one step: one model call, one tool batch, one output check or
    /// one repair prompt, emitting its events through `io`. Final and paused
    /// runs return their outcome again and change nothing.
    pub async fn step(&mut self, io: &dyn AgentIo) -> StepOutcome {
        match &self.state {
            RunState::Done { output } => return StepOutcome::Done(output.clone()),
            RunState::Failed { error } => return StepOutcome::Failed(error.clone()),
            RunState::Paused { reason, .. } => return StepOutcome::Paused(reason.clone()),
            _ => {}
        }
        if self.seq == 0 {
            self.emit(
                io,
                EventBody::RunStarted {
                    workflow: self.spec.persona.clone(),
                },
            );
        }
        self.steps += 1;
        let name = self.state.step_name();
        self.emit(io, EventBody::StepStarted { step: name.into() });
        let mut used = Usage::default();
        let outcome = match self.state.clone() {
            RunState::CallModel => self.call_model(io, &mut used).await,
            RunState::CallTools { calls } => self.call_tools(io, &calls).await,
            RunState::Check { answer } => self.check(io, &answer, &mut used).await,
            RunState::Repair { errors } => {
                self.messages.push(repair_message(&errors));
                self.state = RunState::CallModel;
                StepOutcome::Continue
            }
            RunState::Done { .. } | RunState::Failed { .. } | RunState::Paused { .. } => {
                unreachable!("handled above")
            }
        };
        self.emit(
            io,
            EventBody::StepFinished {
                step: name.into(),
                usage: used,
            },
        );
        match &outcome {
            StepOutcome::Paused(reason) => self.emit(
                io,
                EventBody::RunPaused {
                    reason: reason.clone(),
                },
            ),
            StepOutcome::Done(_) => self.emit(io, EventBody::RunFinished { usage: self.usage }),
            StepOutcome::Failed(error) => self.emit(
                io,
                EventBody::RunFailed {
                    error: error.clone(),
                },
            ),
            StepOutcome::Continue | StepOutcome::Retry { .. } => {}
        }
        outcome
    }

    fn emit(&mut self, io: &dyn AgentIo, body: EventBody) {
        io.events().emit(Event {
            run_id: self.run_id,
            seq: self.seq,
            at_ms: io.clock().now_ms(),
            body,
        });
        self.seq += 1;
    }

    fn fail(&mut self, error: String) -> StepOutcome {
        self.state = RunState::Failed {
            error: error.clone(),
        };
        StepOutcome::Failed(error)
    }

    fn request(&self, io: &dyn AgentIo) -> CompletionRequest {
        let tools = io
            .tools()
            .specs()
            .into_iter()
            .filter(|s| self.spec.tools.contains(&s.name))
            .filter_map(|mut s| {
                if let Some(h) = s.hosted {
                    let used: u64 = self.messages.iter().map(|m| m.hosted_calls(h.name())).sum();
                    let left = u64::from(h.max_uses()).saturating_sub(used);
                    if left == 0 {
                        return None;
                    }
                    s.hosted = Some(
                        h.with_max_uses(u32::try_from(left).expect("left is at most max_uses")),
                    );
                }
                Some(s)
            })
            .collect();
        CompletionRequest {
            model: None,
            system: Some(self.spec.system.clone()),
            messages: self.messages.clone(),
            tools,
            max_tokens: self.spec.max_tokens,
            temperature: None,
            output_schema: self.spec.output.clone(),
        }
    }

    async fn call_model(&mut self, io: &dyn AgentIo, used: &mut Usage) -> StepOutcome {
        if self.model_calls >= self.spec.max_turns {
            return self.fail(format!("max_turns {} reached", self.spec.max_turns));
        }
        let handle = match io.model(&self.spec.model).await {
            Ok(h) => h,
            Err(e) => return self.fail(e.to_string()),
        };
        let req = self.request(io);
        let reply = match self
            .call(io, &handle, &req, true, used, RunState::CallModel)
            .await
        {
            Ok(r) => r,
            Err(outcome) => return outcome,
        };
        self.model_calls += 1;
        let calls: Vec<PendingCall> = reply
            .message
            .tool_calls()
            .map(|(id, name, arguments)| PendingCall {
                id: id.into(),
                name: name.into(),
                arguments: arguments.clone(),
            })
            .collect();
        let answer = reply.message.final_text();
        let kept = reply
            .message
            .parts
            .iter()
            .any(|p| !matches!(p, ContentPart::Text { text } if text.trim().is_empty()));
        if kept {
            self.messages.push(reply.message);
        }
        self.state = if reply.stop == StopReason::PauseTurn {
            RunState::CallModel
        } else if calls.is_empty() {
            RunState::Check { answer }
        } else {
            RunState::CallTools { calls }
        };
        StepOutcome::Continue
    }

    async fn call(
        &mut self,
        io: &dyn AgentIo,
        handle: &ModelHandle,
        req: &CompletionRequest,
        stream: bool,
        used: &mut Usage,
        resume: RunState,
    ) -> Result<CompletionResponse, StepOutcome> {
        let estimate = budget::estimate(req, &handle.price);
        match budget::check(
            &self.limits,
            &handle.price,
            self.usage.cost_usd_micros,
            estimate,
            io.ledger(),
        )
        .await
        {
            Err(e) => {
                return Err(StepOutcome::Retry {
                    error: format!("usage ledger: {e}"),
                    retry_after_ms: None,
                })
            }
            Ok(Some(layer)) => {
                let reason = PauseReason::Budget {
                    limit: layer.into(),
                };
                self.state = RunState::Paused {
                    reason: reason.clone(),
                    resume: Box::new(resume),
                };
                return Err(StepOutcome::Paused(reason));
            }
            Ok(None) => {}
        }
        let result = if stream {
            let (sink, clock, run_id, seq) = (io.events(), io.clock(), self.run_id, &mut self.seq);
            let mut on_delta = |d: &Delta| {
                if let Delta::TextDelta { text } = d {
                    sink.emit(Event {
                        run_id,
                        seq: *seq,
                        at_ms: clock.now_ms(),
                        body: EventBody::TextDelta { text: text.clone() },
                    });
                    *seq += 1;
                }
            };
            handle.model.stream(req, &mut on_delta).await
        } else {
            handle.model.complete(req).await
        };
        let mut reply = match result {
            Ok(r) => r,
            Err(e) if e.is_retryable() => {
                return Err(StepOutcome::Retry {
                    error: e.to_string(),
                    retry_after_ms: e.retry_after_ms,
                })
            }
            Err(e) => return Err(self.fail(e.to_string())),
        };
        reply.usage.cost_usd_micros = handle.price.cost_usd_micros(&reply.usage);
        self.usage += reply.usage;
        *used += reply.usage;
        if let Err(e) = budget::record(&self.limits, io.ledger(), reply.usage).await {
            return Err(self.fail(format!("usage ledger: {e}")));
        }
        Ok(reply)
    }

    async fn call_tools(&mut self, io: &dyn AgentIo, calls: &[PendingCall]) -> StepOutcome {
        assert!(!calls.is_empty(), "a tool step has at least one call");
        for c in calls {
            self.emit(
                io,
                EventBody::ToolCallStarted {
                    call_id: c.id.clone(),
                    name: c.name.clone(),
                },
            );
        }
        let (run_id, step) = (self.run_id, self.steps);
        let allowed = &self.spec.tools;
        let outputs = join_all(calls.iter().enumerate().map(|(i, c)| {
            let ctx = ToolContext::for_call(run_id, step, i);
            async move {
                if !allowed.contains(&c.name) {
                    return ToolOutput::error(format!(
                        "tool `{}` is not allowed for this agent",
                        c.name
                    ));
                }
                ToolOutput::from_result(io.tools().call(&c.name, c.arguments.clone(), &ctx).await)
            }
        }))
        .await;
        assert_eq!(outputs.len(), calls.len());
        let mut parts = Vec::with_capacity(calls.len());
        for (c, out) in calls.iter().zip(outputs) {
            self.emit(
                io,
                EventBody::ToolCallFinished {
                    call_id: c.id.clone(),
                    is_error: out.is_error,
                },
            );
            parts.push(ContentPart::ToolResult {
                call_id: c.id.clone(),
                content: out.content,
                is_error: out.is_error,
            });
        }
        self.messages
            .push(Message::new(Role::Tool, parts).expect("tool results form a valid tool message"));
        self.state = RunState::CallModel;
        StepOutcome::Continue
    }

    async fn check(&mut self, io: &dyn AgentIo, answer: &str, used: &mut Usage) -> StepOutcome {
        let (output, mut errors) = match &self.spec.output {
            None => (Value::String(answer.to_string()), Vec::new()),
            Some(schema) => match parse_json(answer) {
                Err(e) => (Value::Null, vec![e]),
                Ok(v) => {
                    let errs = validate(schema, &v);
                    (v, errs)
                }
            },
        };
        if errors.is_empty() {
            for call in self.spec.evals.clone() {
                if call.name == "judge" {
                    match self.judge(io, &call, &output, answer, used).await {
                        Ok(None) => {}
                        Ok(Some(reason)) => errors.push(format!("judge: {reason}")),
                        Err(outcome) => return outcome,
                    }
                    continue;
                }
                let Some(f) = io.evals().get(&call.name) else {
                    return self.fail(format!("eval `{}` is not registered", call.name));
                };
                if let Err(e) = f(&output, &call.args) {
                    errors.push(format!("{}: {e}", call.name));
                }
            }
        }
        if errors.is_empty() {
            self.state = RunState::Done {
                output: output.clone(),
            };
            return StepOutcome::Done(output);
        }
        if !self.repaired && self.model_calls < self.spec.max_turns {
            self.repaired = true;
            self.state = RunState::Repair { errors };
            return StepOutcome::Continue;
        }
        self.fail(errors.join("; "))
    }

    async fn judge(
        &mut self,
        io: &dyn AgentIo,
        call: &Call,
        output: &Value,
        answer: &str,
        used: &mut Usage,
    ) -> Result<Option<String>, StepOutcome> {
        let criteria = call
            .args
            .first()
            .and_then(Value::as_str)
            .expect("judge criteria checked at persona load");
        let min_score = call
            .args
            .get(1)
            .and_then(Value::as_f64)
            .expect("judge min_score checked at persona load");
        let selector = call
            .args
            .get(2)
            .and_then(Value::as_str)
            .unwrap_or(self.spec.model.as_str())
            .to_string();
        let handle = match io.model(&selector).await {
            Ok(h) => h,
            Err(e) => return Err(self.fail(format!("judge: {e}"))),
        };
        let req = judge_request(criteria, output);
        let resume = RunState::Check {
            answer: answer.to_string(),
        };
        let reply = self.call(io, &handle, &req, false, used, resume).await?;
        Ok(judge_verdict(&reply.message.text_content(), min_score).err())
    }
}

/// Drives `run` until it is done, failed, paused or asks for a retry,
/// saving a checkpoint to `store` before the first step (when the store has
/// none) and after every step. Returns the last outcome.
pub async fn drive(
    run: &mut AgentRun,
    io: &dyn AgentIo,
    store: &dyn CheckpointStore,
) -> Result<StepOutcome, AgentError> {
    if store.latest(run.run_id).await?.is_none() {
        store.save(run.checkpoint()).await?;
    }
    loop {
        let outcome = run.step(io).await;
        store.save(run.checkpoint()).await?;
        if outcome != StepOutcome::Continue {
            return Ok(outcome);
        }
    }
}
