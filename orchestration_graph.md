# Orchestration Graph — Moses + Titan + Ecosystem

> **Version:** 1.0.0  
> **Date:** 2026-06-08  
> **Status:** DRAFT — Round 3 Weaponization  

---

## 1. System Overview

This document is the visual architecture of the coordinated humanoid robotics program. It shows all agents, their roles, data flows, human decision points, fallback paths, and scaling patterns.

**Design Principles (from AGENTS.md):**
- Walker = Primary Orchestrator
- Atlas = Co-Orchestrator (dev domain)
- Cross-silo write requires Tier 3 approval
- Finance data NEVER flows to robotics without per-instance Alex approval
- Silence when something matters is a failure — all agents push proactively

---

## 2. Agent Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ALEX WALK (Human-in-the-Loop)                        │
│                    ┌─────────────────────────────────┐                      │
│                    │  Telegram: Walker L. Boss HQ    │                      │
│                    │  Topics: General, Robotics(10), │                      │
│                    │          Approvals(8), System   │                      │
│                    │          Alerts(9)              │                      │
│                    └─────────────────────────────────┘                      │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │      WALKER (Orchestrator)   │
                    │  • Route messages            │
                    │  • Spawn subagents           │
                    │  • Aggregate status          │
                    │  • Escalation hub            │
                    └──────────────┬──────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼────────┐      ┌──────────▼──────────┐    ┌─────────▼────────┐
│   RENAISSANCE   │      │      ROBOTICS SILO   │    │     ATLAS        │
│   (Finance)     │      │                      │    │   (Dev/CI/CD)    │
│                 │      │  ┌──────────────┐   │    │                  │
│ • DGX budget    │      │  │    TITAN     │   │    │ • Code review    │
│ • Cost tracking │      │  │  Physics/Sim │   │    │ • CI pipelines   │
│ • Envelopes     │◄────►│  └──────┬───────┘   │◄──►│ • Deploy to DGX  │
│ • Alerts        │      │         │           │    │ • Security       │
└─────────────────┘      │  ┌──────▼──────┐   │    └──────────────────┘
                         │  │    MOSES    │   │
                         │  │ Design/Train│   │
                         │  └──────┬──────┘   │
                         └─────────┼──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │      SHARED RESOURCES        │
                    │  /shared/knowledge/robotics/ │
                    │  /shared/sim/titan/          │
                    │  /shared/designs/moses/      │
                    │  /shared/checkpoints/        │
                    └─────────────────────────────┘
```

---

## 3. Data Flow Diagrams

### 3.1 Design → Simulate → Validate Loop (Moses ↔ Titan)

```
┌─────────┐     design_v1.json      ┌─────────┐
│  MOSES  │ ───────────────────────►│  TITAN  │
│         │  (URDF/SDF + scenario)  │         │
│         │                         │ PhysicsEngine
│         │◄────────────────────────│ SensorSimulator
│         │   feasibility_report    │         │
│         │   {FEASIBLE|INFEASIBLE} │         │
│         │                         │         │
│         │  ┌─────────────────┐    │         │
│         │  │ Conflict?       │    │         │
│         │  │ • risk >= 80    │───►│  YES → Tier 3 Gate
│         │  │ • catastrophic  │    │         │
│         │  │ • >2 iterations │    │         │
│         │  └─────────────────┘    │         │
│         │                         │         │
│         │◄── sim_telemetry.jsonl ─┘         │
│         │   (per-timestep state)            │
│         │                                 │
│         │──► Knowledge Corpus ◄───────────┘
│         │    (embeddings + metadata)
└─────────┘
```

**Decision Points:**
- **D1:** Titan says INFEASIBLE → ConflictResolver checks rules → Escalate or Iterate
- **D2:** Sim success rate < 95% + target = physical → Tier 3 gate (PHYSICAL_DEPLOY)
- **D3:** Risk score >= 80 → ALWAYS escalate, no auto-override

### 3.2 Training Job Deploy (Moses ↔ Atlas ↔ Renaissance)

```
┌─────────┐    DEPLOY_REQUEST      ┌─────────┐
│  MOSES  │ ──────────────────────►│  ATLAS  │
│         │  (branch, resources)   │         │
│         │                        │ Build   │
│         │◄───────────────────────│ Loop    │
│         │   DEPLOY_STATUS        │ (0-11)  │
│         │   {BUILDING|TESTING|   │         │
│         │    DEPLOYED|FAILED}    │         │
│         │                        │         │
│         │                        │ ┌───────┴──────┐
│         │                        │ │ atlas-devops │
│         │                        │ │ atlas-testing│
│         │                        │ │ atlas-security
│         │                        │ └──────────────┘
│         │                        │
│         │    COMPUTE_BUDGET      │
│         │    ───────────────────►│
│         │                        │ ┌──────────────┐
│         │◄───────────────────────│ │ RENAISSANCE  │
│         │    BUDGET_RESPONSE     │ │ (envelope)   │
│         │    {approved: true}    │ └──────────────┘
│         │                        │
│         │                        └──────┬───────┘
│         │                               │
│         │                        ┌──────▼───────┐
│         │                        │  DGX Cluster │
│         │                        │  (Slurm/K8s) │
│         │                        └──────────────┘
└─────────┘
```

**Decision Points:**
- **D4:** Budget request > envelope → Tier 3 gate (BUDGET_EXCEED)
- **D5:** Atlas CI gate = RED → Moses fixes, retries (max 3)
- **D6:** DGX queue full → Atlas queues, notifies Moses of ETA

### 3.3 Human-in-the-Loop Flow

```
┌─────────────┐
│   Any Agent │
│  (Moses/    │
│   Titan/    │
│   Atlas)    │
└──────┬──────┘
       │
       │ Detects Tier 3 condition
       ▼
┌─────────────────┐
│ CheckpointEngine│
│ • Evaluate rules│
│ • Create gate   │
└────────┬────────┘
         │
         ├────────────────────────────────────┐
         │                                    │
         ▼                                    ▼
┌─────────────────┐                 ┌─────────────────┐
│ Telegram Topic 8│                 │ Mission Control │
│ (Approvals)     │                 │ Dashboard       │
│                 │                 │                 │
│ ReviewFormatter │                 │ Checkpoint card │
│ • Emoji urgency │                 │ • Progress bar  │
│ • Evidence list │                 │ • Timeout clock │
│ • Reply format  │                 │                 │
└────────┬────────┘                 └─────────────────┘
         │
         ▼
┌─────────────────┐
│   ALEX WALK     │
│                 │
│ Reply options:  │
│ • APPROVE <id>  │
│ • DENY <id>     │
│ • MODIFY <id>   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ process_response│
│ • Update status │
│ • Notify agent  │
│ • Log audit     │
└─────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
 APPROVED   DENIED
    │         │
    ▼         ▼
 Resume    Abort /
 Task      Safe State
```

**Decision Points:**
- **D7:** Timeout reached → safe_default applied (abort / hold / degraded)
- **D8:** Alex replies MODIFY → agent receives modification notes, must re-ack
- **D9:** Emergency override attempted → Check `allow_override` flag, log permanently

---

## 4. Fallback Paths

### 4.1 Titan Failure

```
Titan DOWN
    │
    ├─► 1. Moses falls back to local lightweight sim (Mac Mini M4 Pro)
    │     • Reduced fidelity, faster iteration
    │     • Results marked "local_sim" in corpus
    │
    ├─► 2. If local sim unavailable → Queue designs for Titan recovery
    │     • FIFO queue in /shared/outbox/titan/
    │     • Retry every 5 min, max 1 hour
    │
    └─► 3. If >1 hour down → Walker escalates to Alex
          • Mark robotics silo DEGRADED in Mission Control
          • Suggest manual physics review or external consultant
```

### 4.2 Moses Failure

```
Moses DOWN
    │
    ├─► 1. Titan continues with existing design corpus
    │     • Can re-run sims on prior designs
    │     • No new designs until Moses recovers
    │
    ├─► 2. Atlas pauses Moses CI pipeline
    │     • Other agents unaffected
    │
    └─► 3. Walker spawns recovery subagent
          • Check Moses logs, identify root cause
          • If persistent, alert Alex for manual intervention
```

### 4.3 DGX / Atlas Failure

```
DGX Queue Full or Atlas DOWN
    │
    ├─► 1. Atlas queues job with estimated start time
    │     • Moses notified, can downscale request
    │
    ├─► 2. Fallback to local GPU (Mac Mini or workstation)
    │     • Reduced batch size, longer training
    │     • Results still valid, marked "local_train"
    │
    └─► 3. Cloud burst to GCP/AWS spot instances
          • Renaissance pre-approves spot budget envelope
          • Atlas auto-provisions via Terraform
```

### 4.4 Message Bus Failure

```
Bus DOWN (filesystem unwritable)
    │
    ├─► 1. Agents queue to local outbox
    │     • /shared/outbox/<agent>/ (if shared FS ok)
    │     • ~/.local/openclaw/outbox/ (if total FS failure)
    │
    ├─► 2. Retry every 60s, max 10 attempts
    │
    └─► 3. After 10 failures → Walker alert + Telegram push
          • "Message bus failure — agent coordination impaired"
```

### 4.5 Human Unavailable (Tier 3 Timeout)

```
Alex does not respond to Tier 3 gate
    │
    ├─► Timeout reached (default 24h, configurable per gate)
    │
    ├─► safe_default applied:
    │     • PHYSICAL_DEPLOY → abort
    │     • SAFETY_OVERRIDE → abort
    │     • BUDGET_EXCEED → hold
    │     • DESIGN_BREAKING → hold
    │     • POLICY_UNTESTED → abort
    │     • HARDWARE_PROCURE → hold
    │     • EMERGENCY_STOP → hold
    │     • AGENT_CONFLICT → hold
    │     • CROSS_SILO_WRITE → abort
    │
    └─► Task state logged, Alex notified on next interaction
```

---

## 5. Scaling: Adding More Agents

### 5.1 Current State (June 2026)

```
Agents: 9 LIVE + 2 BUILD MODE + 14 in various stages
Silo count: 5 (personal, bridge, design, bossos, robotics)
Message volume: ~100-500/day (estimated)
```

### 5.2 Scaling Thresholds

| Metric | Current | Threshold | Action |
|--------|---------|-----------|--------|
| Agents | ~25 total | >50 | Shard message bus by silo |
| Msg/sec | <1 | >10 | Add Redis pub/sub layer |
| Sim jobs | ~5/day | >50/day | DGX cluster expansion |
| Checkpoints | ~1/week | >10/day | Dedicated HITL reviewer agent |
| Knowledge corpus | <1GB | >100GB | Vector DB (pgvector / Pinecone) |

### 5.3 Adding a New Agent (e.g., "Hephaestus" — Manufacturing)

```
Step 1: Register in AGENTS.md
        ID: hephaestus, Silo: robotics, Model: Claude Sonnet 4.6

Step 2: Generate Ed25519 keypair, store in Vault

Step 3: Create inbox/outbox directories
        /shared/inbox/hephaestus/
        /shared/outbox/hephaestus/

Step 4: Subscribe to message types
        DESIGN_SUBMISSION (from Moses) → check manufacturability
        FEASIBILITY_REPORT (from Titan) → check buildability

Step 5: Define tier rules
        Tier 2: Design feedback, DFM analysis
        Tier 3: CNC program deploy, supplier order > $500

Step 6: Add to Mission Control
        Health endpoint: /health/hephaestus
        Status page tab: Manufacturing

Step 7: Budget envelope
        Renaissance allocates monthly spend cap

Step 8: Onboarding test
        Send test message, verify end-to-end in < 5 min
```

### 5.4 Silo Expansion

```
Current Silos:
  personal  → Renaissance, Raja
  bridge    → Bridge Ops, Crello
  design    → Jonny
  bossos    → Atlas + moons
  robotics  → Titan, Moses

Future Silos (proposed):
  manufacturing → Hephaestus
  legal         → Advocatus
  security      → CY
  science       → Gravity, Qubit

Cross-silo rules (from AGENTS.md):
  • Read-only: any → any (with logging)
  • Write: requires Tier 3 + explicit Alex approval
  • Finance → any: aggregate only, no positions
  • BossOS code → any: NEVER (L2 CONFIDENTIAL)
```

---

## 6. Mission Control Integration

All status flows to **Walker HQ at http://100.85.182.115:8080** (per AGENTS.md).

### New Tabs Required

| Tab | Data Source | Update Frequency |
|-----|-------------|------------------|
| Robotics Pipeline | Moses + Titan | Real-time (WebSocket) |
| Sim Queue | Titan | Every 30s |
| DGX Jobs | Atlas + Renaissance | Every 60s |
| Tier 3 Gates | CheckpointEngine | On change |
| Knowledge Graph | Corpus | Daily batch |
| Agent Health | All agents | Every 5 min |

### Alert Rules

| Condition | Severity | Channel |
|-----------|----------|---------|
| Titan sim crash | WARNING | Telegram Robotics topic |
| Moses build fail | WARNING | Telegram Robotics topic |
| Tier 3 gate created | INFO | Telegram Approvals topic |
| Tier 3 gate timeout | CRITICAL | Telegram + Email |
| DGX budget 80% spent | WARNING | Telegram Finance topic |
| DGX budget exceeded | CRITICAL | Telegram + Email + PagerDuty |
| Agent heartbeat lost | CRITICAL | Telegram System Alerts |

---

## 7. Decision Points Summary

| ID | Location | Condition | Human Required? | Safe Default |
|----|----------|-----------|-----------------|--------------|
| D1 | Moses ↔ Titan | Titan INFEASIBLE + conflict | Maybe (resolver) | Iterate or Escalate |
| D2 | Moses ↔ Titan | sim_success < 95% + physical target | YES | Abort |
| D3 | Titan analysis | risk_score >= 80 | YES | Abort |
| D4 | Moses ↔ Renaissance | projected_spend > 110% approved | YES | Hold |
| D5 | Atlas CI | Gate = RED | No (auto-retry ×3) | Abort after 3 |
| D6 | Atlas ↔ DGX | Queue full | No (auto-queue) | Local fallback |
| D7 | HITL | Timeout | N/A (system) | Per gate rule |
| D8 | HITL | Alex replies MODIFY | Yes (re-ack) | Hold until ack |
| D9 | HITL | Override attempted | Yes (if allowed) | Log + notify |

---

*Orchestration Graph v1.0 — Boss Industries — 2026-06-08*
