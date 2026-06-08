# Moses Integration Specification

> **Version:** 1.0.0  
> **Date:** 2026-06-08  
> **Status:** DRAFT — Round 3 Weaponization  
> **Classification:** Internal — Boss Industries  

---

## 1. Objective

Define how **Moses** (humanoid robotics weaponized builder) integrates with the existing Walker agent ecosystem: Titan, Walker, Atlas, Renaissance, and the cross-agent message bus. This spec is the source of truth for APIs, data flows, error handling, and scaling.

---

## 2. Ecosystem Context (Observed)

| Agent | Role | Silo | Status | Relevant APIs |
|-------|------|------|--------|---------------|
| **Walker** | Chief of Staff / Orchestrator | — | LIVE | Telegram routing, session spawn, cron |
| **Titan** | Physics / Simulation / Safety | robotics | v3.0 BUILD MODE | `simulation.py`, `PhysicsEngine`, `SensorSimulator` |
| **Atlas** | Code / CI/CD / DevOps | bossos | LIVE | Git, PRs, Claude Code, Supabase, moons |
| **Renaissance** | Finance / Budget / Compute | personal | WEAPONIZED | Brevo SMTP, cost tracking, DGX budget |
| **Moses** | Design / CAD / Policy Training | robotics | v1.0 BUILD MODE | **NEW** — this spec defines its interfaces |

**AGENTS.md Routing Rules Applied:**
- Telegram Topic 10 = Robotics → Titan + Moses
- Cross-silo write requires Tier 3 approval
- Walker = primary orchestrator; Atlas = co-orchestrator for dev domain
- Finance data NEVER flows to robotics without per-instance Alex approval

---

## 3. Integration Topologies

### 3.1 Moses ↔ Titan

**Purpose:** Design-simulate loop. Moses designs; Titan validates physics and safety.

**Data Flow:**
```
Moses                          Titan
  │                              │
  ├─► design_v1.json ───────────►├─ Load into PhysicsEngine
  │    (URDF/SDF/JSON)           │   run_simulation()
  │                              │
  │◄─ feasibility_report.json ───┤
  │    {status: FEASIBLE |      │
  │     INFEASIBLE | NEEDS_WORK,│
  │     constraints: [...],      │
  │     risk_score: 0-100}       │
  │                              │
  ├─► design_v2.json ───────────►├─ Iterate...
  │                              │
  │◄─ sim_telemetry.jsonl ───────┤
  │    (state per timestep)      │
```

**Shared Knowledge Corpus:**
- Location: `/shared/knowledge/robotics/corpus/`
- Format: JSONL with embeddings (OpenAI `text-embedding-3-large`)
- Sync: Titan writes sim results; Moses reads for policy training
- Conflict resolution: see `titan_moses_protocol.py`

**API Schema — Moses → Titan:**
```json
{
  "message_type": "DESIGN_SUBMISSION",
  "message_id": "moses-20260608-001",
  "from": "moses",
  "to": "titan",
  "payload": {
    "design": {
      "format": "urdf|sdf|onshape_url|mjcf",
      "uri": "/shared/designs/moses/humanoid_v1.urdf",
      "checksum": "sha256:abc123..."
    },
    "test_scenario": {
      "type": "walk|manipulate|balance|fall_recovery",
      "duration_sec": 60,
      "terrain": "flat|slope|stairs|uneven",
      "perturbations": ["push_50N_0.5s", "trip_left_ankle"]
    },
    "priority": "normal|urgent|blocking",
    "deadline": "2026-06-09T12:00:00Z"
  },
  "tier": 2
}
```

**API Schema — Titan → Moses:**
```json
{
  "message_type": "FEASIBILITY_REPORT",
  "in_reply_to": "moses-20260608-001",
  "from": "titan",
  "to": "moses",
  "payload": {
    "status": "FEASIBLE|INFEASIBLE|NEEDS_WORK|SIM_ERROR",
    "physics_validation": {
      "com_position": [0.12, 0.0, 0.85],
      "stability_margin": 0.04,
      "pass": false
    },
    "safety_analysis": {
      "risk_score": 73,
      "max_joint_torque": 120.5,
      "collision_probability": 0.15,
      "fail_modes": ["ankle_roll_overload", "knee_hyperextension"]
    },
    "recommendations": [
      {
        "severity": "critical|warning|info",
        "component": "left_ankle_pitch",
        "issue": "Torque exceeds servo spec by 18%",
        "suggestion": "Reduce shank mass or increase gear ratio"
      }
    ],
    "sim_artifacts": {
      "telemetry": "/shared/sim/titan/moses-20260608-001/telemetry.jsonl",
      "video": "/shared/sim/titan/moses-20260608-001/render.mp4",
      "log": "/shared/sim/titan/moses-20260608-001/sim.log"
    }
  },
  "tier": 2
}
```

**Error Handling:**
- Titan SIM_ERROR → Moses retries with simplified design (max 3 retries)
- Titan timeout (>30 min for 60s sim) → Escalate to Walker, queue on DGX if available
- Checksum mismatch → Reject immediately, request re-upload

---

### 3.2 Moses ↔ Walker

**Purpose:** Reporting, escalation, orchestration of multi-agent tasks.

**Data Flow:**
```
Moses                          Walker
  │                              │
  ├─► status_report.json ───────►├─ Aggregate in Mission Control
  │    (daily build status)      │
  │                              │
  ├─► ESCALATION ───────────────►├─ Tier 3 gate triggered
  │    (human checkpoint req)    │   → Route to Approvals topic
  │                              │
  │◄─ orchestration_cmd.json ────┤
  │    (spawn subagent,          │
  │     reallocate compute)      │
```

**API Schema — Moses → Walker (Status):**
```json
{
  "message_type": "AGENT_STATUS",
  "from": "moses",
  "to": "walker",
  "payload": {
    "agent_state": "building|testing|waiting_human|error",
    "current_task": "Training policy v3 on DGX",
    "progress_pct": 67,
    "artifacts_produced": [
      {"type": "urdf", "path": "...", "size_bytes": 45023}
    ],
    "compute_used": {
      "dgx_gpu_hours": 12.5,
      "estimated_cost_usd": 47.30
    },
    "blockers": [],
    "next_milestone": "2026-06-08T18:00:00Z"
  }
}
```

**API Schema — Moses → Walker (Escalation):**
```json
{
  "message_type": "ESCALATION",
  "from": "moses",
  "to": "walker",
  "payload": {
    "escalation_type": "TIER_3_REQUIRED|AGENT_FAILURE|BUDGET_EXCEEDED|CONFLICT",
    "reason": "Titan declared INFEASIBLE on knee design; Moses believes test is worth running",
    "context": {
      "titan_report": "/shared/sim/titan/moses-20260608-001/report.json",
      "moses_counter": "/shared/designs/moses/counter_argument.md"
    },
    "requested_action": "HUMAN_REVIEW",
    "urgency": "normal|urgent"
  }
}
```

**Error Handling:**
- Walker unreachable → Moses queues to local filesystem (`/shared/outbox/walker/`), retries every 5 min
- Escalation unacknowledged >24h → Moses aborts task, logs to `.learnings/ERRORS.md`

---

### 3.3 Moses ↔ Atlas

**Purpose:** Code review, CI/CD, deployment of Moses software artifacts.

**Data Flow:**
```
Moses                          Atlas
  │                              │
  ├─► git push ─────────────────►├─ Trigger CI pipeline
  │    (feature/moses-v1.2)      │   atlas-devops runs tests
  │                              │
  │◄─ PR review ─────────────────┤
  │    (atlas-security SAST)     │
  │                              │
  ├─► deploy request ───────────►├─ Tier 3 gate
  │    (DGX training job)        │   → Atlas deploys to Slurm/K8s
```

**API Schema — Moses → Atlas (Deploy Request):**
```json
{
  "message_type": "DEPLOY_REQUEST",
  "from": "moses",
  "to": "atlas",
  "payload": {
    "repo": "boss-industries/moses",
    "branch": "feature/policy-v3",
    "commit": "a1b2c3d...",
    "target": {
      "environment": "dgx-cluster|local-gpu|sim-cluster",
      "job_type": "training|inference|batch_eval",
      "resources": {
        "gpus": 4,
        "cpu_cores": 32,
        "ram_gb": 128,
        "duration_hours": 24
      }
    },
    "artifacts_to_produce": [
      "policy.pt", "training_curves.png", "eval_report.json"
    ],
    "tier": 3
  }
}
```

**Atlas Build Loop Integration:**
- Atlas runs Steps 0–11 of its Build Loop on Moses code
- `atlas-security` moon spawned on auth/data-exposure changes
- `atlas-testing` moon spawned on new training eval code
- Gate must be GREEN or YELLOW before deploy

**Error Handling:**
- CI fails → Atlas returns diagnostic; Moses fixes and resubmits (max 3 retries)
- DGX queue full → Atlas queues job, notifies Moses of estimated start time
- Deploy timeout → Atlas kills job, preserves partial artifacts, alerts Walker

---

### 3.4 Moses ↔ Renaissance

**Purpose:** Budget tracking for DGX compute and robotics hardware procurement.

**Data Flow:**
```
Moses                          Renaissance
  │                              │
  ├─► compute_request.json ─────►├─ Check budget envelope
  │    ($500 for 48h DGX)        │
  │                              │
  │◄─ budget_approval.json ──────┤
  │    {approved: true,          │
  │     envelope_id: "dgx-06"}   │
  │                              │
  ├─► actual_spend.json ────────►├─ Track burn
  │    (hourly telemetry)        │
```

**API Schema — Moses → Renaissance:**
```json
{
  "message_type": "COMPUTE_BUDGET_REQUEST",
  "from": "moses",
  "to": "renaissance",
  "payload": {
    "project": "moses-humanoid-v1",
    "request_type": "dgx_gpu|cloud_storage|hardware_procurement",
    "amount_usd": 500.00,
    "justification": "Training locomotion policy v3 on 4x A100 for 48h",
    "expected_output": "policy.pt with >85% success rate on flat terrain",
    "duration_hours": 48,
    "priority": "normal|urgent"
  }
}
```

**API Schema — Renaissance → Moses:**
```json
{
  "message_type": "BUDGET_RESPONSE",
  "from": "renaissance",
  "to": "moses",
  "payload": {
    "approved": true|false,
    "envelope_id": "dgx-2026-06",
    "approved_amount_usd": 500.00,
    "conditions": [
      "Auto-terminate if cost exceeds 110% of approved",
      "Require eval report before next request"
    ],
    "remaining_budget_usd": 4200.00,
    "month_to_date_spend_usd": 1800.00
  }
}
```

**Error Handling:**
- Budget denied → Moses downscales request or queues for next month
- Spend exceeds 110% → Renaissance sends KILL signal to Atlas; job terminated
- Renaissance unreachable → Moses uses last-known envelope with 10% safety margin

---

## 4. Cross-Agent Message Bus Protocol

### 4.1 Architecture

Hybrid design: **Filesystem for durability** + **in-memory bus for real-time**.

```
┌─────────────────────────────────────────────────────────────┐
│                    MESSAGE BUS LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  Real-Time Bus (in-memory)   │  Durable Store (filesystem)  │
│  • Redis pub/sub (optional)  │  • /shared/inbox/<agent>/    │
│  • Asyncio queues (default)  │  • /shared/outbox/<agent>/   │
│  • Latency: <1ms             │  • Retention: 90 days        │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Message Schema (Universal Envelope)

Every inter-agent message MUST use this envelope:

```json
{
  "envelope_version": "1.0",
  "message_id": "uuid-v4",
  "correlation_id": "uuid-v4|null",
  "timestamp_utc": "2026-06-08T14:30:00Z",
  "from": "agent_id",
  "to": "agent_id|broadcast",
  "message_type": "TYPE",
  "priority": 1|2|3|4|5,
  "ttl_seconds": 86400,
  "payload": { ... },
  "signature": {
    "algo": "ed25519",
    "value": "base64..."
  }
}
```

**Message Types Registry:**

| Type | Direction | Purpose |
|------|-----------|---------|
| `DESIGN_SUBMISSION` | Moses → Titan | Send design for sim |
| `FEASIBILITY_REPORT` | Titan → Moses | Sim results |
| `AGENT_STATUS` | Any → Walker | Heartbeat / progress |
| `ESCALATION` | Any → Walker | Human review needed |
| `DEPLOY_REQUEST` | Moses → Atlas | CI/CD / training job |
| `DEPLOY_STATUS` | Atlas → Moses | Build / deploy progress |
| `COMPUTE_BUDGET_REQUEST` | Moses → Renaissance | Ask for $ |
| `BUDGET_RESPONSE` | Renaissance → Moses | Approve / deny |
| `HUMAN_CHECKPOINT` | Any → Approvals Topic | Tier 3 gate |
| `HUMAN_RESPONSE` | Alex → Any | Approval / deny / modify |
| `KNOWLEDGE_UPDATE` | Any → Corpus | New data to shared KG |
| `SYSTEM_ALERT` | Any → Broadcast | Health, security, cost |

### 4.3 Routing Rules

1. **Direct:** `to` field specifies single agent → deliver to `/shared/inbox/<agent>/`
2. **Broadcast:** `to: "broadcast"` → deliver to all agents' inboxes
3. **Topic:** Messages with `message_type: HUMAN_CHECKPOINT` → also copied to Telegram Approvals topic (Thread 8)
4. **Priority:** 1 = emergency (immediate push to Alex), 5 = background (batch processing)

### 4.4 Delivery Guarantees

| Guarantee | Mechanism | Fallback |
|-----------|-----------|----------|
| At-least-once | Filesystem inbox + idempotency keys | Retry every 60s, max 10 |
| Ordering | Monotonic timestamp per sender | NTP sync required |
| Durability | Write to tmp, fsync, rename | Log to `.learnings/ERRORS.md` |

### 4.5 Security

- All messages signed with agent Ed25519 key (stored in Vault)
- Cross-silo messages encrypted with recipient public key
- `signature` verified before processing
- Failed verification → quarantine in `/shared/quarantine/`

---

## 5. Error Handling Matrix

| Failure Mode | Detection | Response | Escalation |
|--------------|-----------|----------|------------|
| Titan sim crash | Exit code != 0 | Moses retries simplified design ×3 | Walker after 3rd fail |
| Moses design corrupt | Checksum mismatch | Reject, request re-upload | — |
| Atlas CI fail | Test exit code != 0 | Return diagnostic, Moses fixes | Walker if 3 retries fail |
| DGX job timeout | >110% budget or >deadline | Atlas kills job, preserves artifacts | Renaissance + Walker |
| Budget exceeded | Renaissance monitor | Kill job, alert Alex | Immediate |
| Message bus down | Inbox not writable | Queue to local outbox, retry 60s | Walker after 10 min |
| Agent unreachable | No heartbeat 5 min | Mark DEGRADED in Mission Control | Walker |
| Human checkpoint timeout | >24h no response | Abort task, safe state | Walker logs to ERRORS.md |

---

## 6. Scaling: Adding N More Agents

### 6.1 Agent Onboarding Checklist

1. **Register in AGENTS.md** — ID, name, silo, model tiers
2. **Generate keypair** — Ed25519, store in Vault
3. **Create inbox/outbox** — `/shared/inbox/<agent>/`, `/shared/outbox/<agent>/`
4. **Subscribe to message types** — Add to bus routing table
5. **Define tier rules** — Which ops need Tier 2 vs Tier 3
6. **Add to Mission Control** — Health endpoint, status page
7. **Budget envelope** — Renaissance allocates monthly spend cap

### 6.2 Message Bus Scaling

- **<10 agents:** Filesystem-based bus is sufficient
- **10–50 agents:** Add Redis pub/sub layer, keep filesystem as audit log
- **>50 agents:** Shard by silo, add message broker (RabbitMQ / NATS)

### 6.3 Compute Scaling

- DGX cluster: Slurm scheduler, queue-based
- Cloud burst: GCP / AWS spot instances for sim
- Local fallback: Mac Mini M4 Pro for small sims (per Skywalker model)

---

## 7. File Locations

| File | Path |
|------|------|
| This spec | `workspace/moses-integration/integration_spec.md` |
| Titan-Moses protocol | `workspace/moses-integration/titan_moses_protocol.py` |
| Human checkpoint | `workspace/moses-integration/human_checkpoint.py` |
| Orchestration graph | `workspace/moses-integration/orchestration_graph.md` |
| Activation sequence | `workspace/moses-integration/activation_sequence.md` |
| Shared inbox | `/shared/inbox/<agent_id>/` |
| Shared outbox | `/shared/outbox/<agent_id>/` |
| Shared knowledge corpus | `/shared/knowledge/robotics/corpus/` |
| Sim artifacts | `/shared/sim/titan/<message_id>/` |
| Design artifacts | `/shared/designs/moses/` |

---

## 8. Validation Gates

- [ ] All APIs have concrete JSON schemas (defined above)
- [ ] Error handling defined for every integration point
- [ ] Tier 3 gates identified for physical deploy, budget, safety overrides
- [ ] Message bus supports at-least-once delivery
- [ ] Scaling path to 10+ agents is explicit
- [ ] AGENTS.md routing rules referenced accurately

---

*Integration Spec v1.0 — Boss Industries — 2026-06-08*
