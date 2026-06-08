# Activation Sequence — Moses + Titan Ecosystem Integration

> **Version:** 1.0.0  
> **Date:** 2026-06-08  
> **Status:** DRAFT — Round 3 Weaponization  

---

## 0. Prerequisites

Before starting activation, verify:

| Item | Status | Verification Command |
|------|--------|---------------------|
| OpenClaw gateway running | REQUIRED | `openclaw status` |
| Walker HQ (Mission Control) accessible | REQUIRED | `curl http://100.85.182.115:8080/health` |
| Telegram bots operational | REQUIRED | Send test message in General topic |
| Titan v3.0 code present | REQUIRED | `ls /Users/aiagent/.openclaw/workspace/future-agents/titan-agent/` |
| DGX cluster access configured | REQUIRED | `ssh dgx 'nvidia-smi -L'` |
| Vault secrets accessible | REQUIRED | `openclaw vault list secret/walker/` |
| Shared filesystem mounted | REQUIRED | `ls /shared/` |
| Renaissance budget envelope defined | REQUIRED | Check `renaissance/AGENT.md` §Integration Points |

---

## Phase 1: Day 1 — Install, Configure, First Run

**Goal:** Moses and Titan can exchange a single design + simulation message end-to-end.

### 1.1 Create Integration Directory Structure

```bash
# Run as aiagent user
mkdir -p /shared/{inbox,outbox}/{moses,titan,walker,atlas,renaissance}
mkdir -p /shared/{sim/titan,designs/moses,knowledge/robotics/corpus,checkpoints}
mkdir -p /shared/outbox/{moses,titan,walker,atlas,renaissance}/archive
chmod 755 /shared
```

**What can go wrong:**
- `/shared` not mounted → Check fstab or NFS config. Fallback: use `~/.openclaw/shared/` locally.
- Permission denied → Ensure `aiagent` user owns `/shared` or is in correct group.

### 1.2 Install Integration Artifacts

```bash
cd /Users/aiagent/.openclaw/workspace
mkdir -p moses-integration
cp integration_spec.md moses-integration/
cp titan_moses_protocol.py moses-integration/
cp human_checkpoint.py moses-integration/
cp orchestration_graph.md moses-integration/
cp activation_sequence.md moses-integration/
```

### 1.3 Configure Agent Keys

```bash
# Generate Ed25519 keypairs for message signing
# (In production, use OpenClaw Vault; for Day 1, local files are acceptable)

python3 -c "
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import base64
for agent in ['moses', 'titan']:
    key = Ed25519PrivateKey.generate()
    priv = base64.b64encode(key.private_bytes_raw()).decode()
    pub = base64.b64encode(key.public_key().public_bytes_raw()).decode()
    open(f'/shared/.keys/{agent}.pub', 'w').write(pub)
    open(f'/shared/.keys/{agent}.priv', 'w').write(priv)
    print(f'{agent}: ok')
"
chmod 600 /shared/.keys/*.priv
```

**What can go wrong:**
- `cryptography` not installed → `pip install cryptography`
- Keys lost → Regenerate; update all agents' key caches

### 1.4 Bootstrap Titan v3.0 Simulation Environment

```bash
cd /Users/aiagent/.openclaw/workspace/future-agents/titan-agent

# Verify simulation.py works
python3 scripts/simulation.py
# Expected output: Simulation complete! Final state: 4 objects, Collisions: 0

# Install dependencies
pip install -r requirements.txt

# Create default scenario for Moses testing
python3 -c "
import json
from scripts.simulation import create_default_scenario
with open('/shared/sim/titan/default_scenario.json', 'w') as f:
    json.dump(create_default_scenario(), f, indent=2)
"
```

**What can go wrong:**
- `simulation.py` crashes → Check Python version (requires 3.10+), check asyncio event loop
- Dependencies missing → Install individually, pin versions in `requirements.txt`

### 1.5 Bootstrap Moses Agent Skeleton

```bash
mkdir -p /Users/aiagent/.openclaw/workspace/moses-agent/{designs,policies,scripts,logs}

# Create minimal AGENT.md for Moses
cat > /Users/aiagent/.openclaw/workspace/moses-agent/AGENT.md << 'EOF'
# AGENT.md — MOSES Operating Manual
# Humanoid Robotics Weaponized Builder
# Silo: robotics
# Reports to: Walker
# Coordinates with: Titan, Atlas, Renaissance

## Identity
- ID: moses
- Name: MOSES (Modular Operational System for Embodied Simulation)
- Bot: @WalkerMosesBot
- Primary Model: Claude Sonnet 4.6
- Fallback: Claude Opus 4.6

## Routing
- Telegram Topic 10 (Robotics)
- Patterns: humanoid, biped, locomotion, manipulation, policy, URDF, MuJoCo, Isaac

## Tools
- `design_cad` → Generate/modify URDF/SDF designs
- `train_policy` → RL training job submission
- `submit_to_titan` → Send design for physics validation
- `request_budget` → Ask Renaissance for compute allocation

## Safety
- All physical deploys require Tier 3 approval
- Designs must pass Titan feasibility before training
- Budget envelope enforced by Renaissance
EOF
```

### 1.6 First End-to-End Test

```bash
cd /Users/aiagent/.openclaw/workspace/moses-integration
python3 titan_moses_protocol.py
```

**Expected output:**
```
[Moses] sent DESIGN_SUBMISSION → titan (uuid...)
[Titan] received design /shared/designs/moses/demo_arm.urdf
=== RESULT ===
Action: ACCEPT
Reason: Titan verdict: FEASIBLE. No counter-argument.
```

**What can go wrong:**
- Message not delivered → Check `/shared/outbox/titan/` for `.json` file; check permissions
- Titan doesn't process → Ensure `titan_moses_protocol.py` demo runs both sides (Moses + Titan mock)
- Checksum mismatch → Verify file wasn't modified after checksum generation

### 1.7 Verify Message Bus

```bash
# Check that messages are durable
ls -la /shared/outbox/titan/
ls -la /shared/inbox/moses/

# Verify archive works
ls -la /shared/inbox/moses/archive/
```

### Day 1 Validation Gate

- [ ] Directory structure created
- [ ] `titan_moses_protocol.py` demo runs successfully
- [ ] Message bus delivers and archives messages
- [ ] Titan simulation.py executes without error
- [ ] Moses AGENT.md created and registered

---

## Phase 2: Week 1 — Full Loop Operational

**Goal:** Moses can design, Titan can simulate, Atlas can build, Renaissance can track budget — all in one coordinated loop.

### 2.1 Integrate with Atlas CI/CD

```bash
# Atlas must recognize moses-agent repo
cd /Users/aiagent/.openclaw/workspace/moses-agent
git init
git remote add origin https://github.com/walkerlboss-dot/moses-agent.git

# Create initial structure
touch README.md
mkdir -p src/{design,sim,train,deploy}
touch src/__init__.py

# First commit
git add -A
git commit -m "feat: moses agent v0.1 bootstrap"
```

**Atlas Build Loop integration:**
- Atlas monitors `moses-agent` repo
- On push to `feature/*`, Atlas runs Steps 0–11
- `atlas-security` moon spawned on auth changes
- `atlas-testing` moon spawned on new eval code

**What can go wrong:**
- Atlas repo size limit (6.2GB caused timeouts before) → Use SSH instead of HTTPS, or push off-peak
- Tests fail on baseline → Document pre-existing failures as YELLOW gate, don't fix inside task

### 2.2 Integrate with Renaissance Budget Tracking

```bash
# Create budget envelope for Moses
# (Renaissance operator: run this)
cd /Users/aiagent/.openclaw/workspace/renaissance
python3 -c "
import json
envelope = {
    'envelope_id': 'dgx-2026-06-moses',
    'project': 'moses-humanoid-v1',
    'monthly_budget_usd': 5000.00,
    'allocated_to': 'moses',
    'alert_threshold_pct': 80,
    'kill_threshold_pct': 110,
    'created_at': '2026-06-08T00:00:00Z'
}
with open('budgets/dgx-2026-06-moses.json', 'w') as f:
    json.dump(envelope, f, indent=2)
print('Envelope created')
"
```

**What can go wrong:**
- Renaissance not running → Start `renaissance_cli.py` or check cron
- Budget file not found → Ensure `budgets/` directory exists in Renaissance workspace

### 2.3 Configure Human Checkpoint System

```bash
cd /Users/aiagent/.openclaw/workspace/moses-integration
python3 -c "
from human_checkpoint import CheckpointEngine, GateType, Urgency
from pathlib import Path

engine = CheckpointEngine()
print('CheckpointEngine initialized')
print('Active checkpoints:', engine.get_summary())
"
```

**Test a Tier 3 gate:**
```bash
python3 -c "
from human_checkpoint import HITLGuard, GateType, Urgency
from pathlib import Path

guard = HITLGuard()
cp = guard.check_before_action(
    action_type=GateType.PHYSICAL_DEPLOY,
    context={'target_environment': 'physical', 'sim_success_rate': 87.0},
    evidence={'sim_report.json': Path('/tmp/fake.json')},
    requested_by='moses',
    urgency=Urgency.URGENT
)
if cp:
    print(f'Gate created: {cp.checkpoint_id}')
    print(f'Status: {cp.status.value}')
    print(f'Timeout: {cp.timeout_at}')
else:
    print('No gate triggered (unexpected)')
"
```

**What can go wrong:**
- Checkpoint not saved → Check `/shared/checkpoints/` permissions
- Rule not triggered → Verify context keys match rule `auto_conditions`

### 2.4 Full Loop Test

```
Moses                    Titan                   Atlas              Renaissance
  │                        │                       │                    │
  ├─ design arm_v1.urdf ──►│                       │                    │
  │                        ├─ run sim ─────────────┤                    │
  │◄─ FEASIBLE ────────────┤                       │                    │
  │                        │                       │                    │
  ├─ train policy v1 ─────────────────────────────►├─ request budget ──►│
  │                        │                       │◄─ approved ────────┤
  │                        │                       │                    │
  │◄─ DEPLOY_STATUS: DONE ─┤                       │                    │
```

**Validation:**
- All four agents exchanged messages
- Budget tracked in Renaissance
- Checkpoint created if sim_success < 95% + physical target

### Week 1 Validation Gate

- [ ] Moses repo in git with Atlas CI triggered
- [ ] Renaissance budget envelope created and responsive
- [ ] HITL gate created and resolved via test
- [ ] Full loop test completed (design → sim → train → budget)
- [ ] All errors logged to `.learnings/ERRORS.md`

---

## Phase 3: Month 1 — Multi-Agent Coordination Live

**Goal:** All agents operate autonomously with human oversight at Tier 3 gates only. System runs 24/7.

### 3.1 Cron and Heartbeat Setup

```bash
# Moses heartbeat — every 15 minutes
# (Add to crontab after surgery mode ends)
*/15 * * * * cd /Users/aiagent/.openclaw/workspace/moses-agent && python3 scripts/heartbeat.py >> logs/heartbeat.log 2>&1

# Titan sim queue processor — every 5 minutes
*/5 * * * * cd /Users/aiagent/.openclaw/workspace/future-agents/titan-agent && python3 scripts/process_queue.py >> logs/queue.log 2>&1

# Checkpoint timeout checker — every 10 minutes
*/10 * * * * cd /Users/aiagent/.openclaw/workspace/moses-integration && python3 -c "from human_checkpoint import CheckpointEngine; CheckpointEngine().check_timeouts()" >> logs/hitl.log 2>&1

# Renaissance budget sync — hourly
0 * * * * cd /Users/aiagent/.openclaw/workspace/renaissance && python3 scripts/budget_sync.py >> logs/budget.log 2>&1
```

**What can go wrong:**
- Cron jobs overlap → Use flock or systemd timers
- Log disk fills → Set up logrotate
- Surgery mode still active → Alex must explicitly re-enable crons

### 3.2 Mission Control Integration

Add to `~/.openclaw/walker-mission-control/public/hub.html`:

```html
<!-- Robotics Tab -->
<div id="robotics-tab" class="tab-pane">
  <h2>🤖 Robotics Pipeline</h2>
  <div id="moses-status">Loading...</div>
  <div id="titan-queue">Loading...</div>
  <div id="dgx-jobs">Loading...</div>
  <div id="tier3-gates">Loading...</div>
</div>
```

Add API endpoints to server.js:
```javascript
app.get('/api/robotics/status', (req, res) => {
  // Read from /shared/checkpoints/ and agent inboxes
  res.json({ moses: 'building', titan: 'idle', dgx: 2 });
});
```

### 3.3 Knowledge Corpus Growth

```bash
# Weekly embedding computation
# (Run as background job)
cd /shared/knowledge/robotics/corpus
python3 -c "
from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer('all-MiniLM-L6-v2')
with open('index.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        text = json.dumps(entry['content'])
        embedding = model.encode(text).tolist()
        entry['embedding'] = embedding
        # Write to vector DB or append back
"
```

**What can go wrong:**
- Corpus grows too large → Shard by month, archive old entries
- Embedding model unavailable → Fallback to OpenAI API
- Vector DB not set up → Use pgvector in Supabase (CROWN)

### 3.4 Stress Test

Simulate high load:
```bash
# Submit 10 designs in rapid succession
for i in {1..10}; do
  python3 -c "
from titan_moses_protocol import DesignArtifact, TestScenario, TitanInterface, FilesystemMessageBus, KnowledgeCorpus
from pathlib import Path
bus = FilesystemMessageBus('moses')
corpus = KnowledgeCorpus()
titan = TitanInterface(bus, corpus)
d = DesignArtifact('urdf', Path('/shared/designs/moses/stress_$i.urdf'), 'fake')
d.uri.write_text('<robot/>')
titan.submit_design(d, TestScenario('walk'))
"
done
```

Monitor:
- Message bus latency (should be < 100ms for filesystem)
- Titan queue depth (should clear within 30 min)
- Checkpoint creation rate (should not exceed 10/day in normal ops)

### 3.5 Documentation and Handoff

Update living documents:
- `integration_spec.md` — API versions, schema changes
- `titan_moses_protocol.py` — Version bump, changelog
- `human_checkpoint.py` — New rules as discovered
- `orchestration_graph.md` — Agent additions, flow changes
- `AGENTS.md` — Moses and Titan v3.0 moved to LIVE roster

### Month 1 Validation Gate

- [ ] Cron jobs running 24/7 without error
- [ ] Mission Control shows real-time robotics status
- [ ] Knowledge corpus > 100 entries with embeddings
- [ ] Stress test passed (10 concurrent designs)
- [ ] Zero unacknowledged Tier 3 gates > 24h
- [ ] All agents report healthy heartbeat
- [ ] Documentation updated and reviewed

---

## Phase 4: Failure Modes and Recovery

### 4.1 Common Failures by Phase

| Phase | Common Failure | Root Cause | Recovery |
|-------|---------------|------------|----------|
| Day 1 | `/shared` not writable | NFS mount failed | Use local fallback `~/.openclaw/shared/` |
| Day 1 | Key generation fails | Missing `cryptography` | `pip install cryptography` |
| Day 1 | Titan sim crashes | Python version < 3.10 | Upgrade Python or use conda env |
| Week 1 | Atlas CI timeout | Repo too large | Use SSH, push off-peak, or LFS |
| Week 1 | Budget not found | Renaissance not started | Start `renaissance_cli.py` |
| Week 1 | Gate not triggered | Context keys mismatch | Debug with `CheckpointEngine.evaluate_conditions()` |
| Month 1 | Cron overlap | Jobs take longer than interval | Use `flock` or increase interval |
| Month 1 | Disk full | Logs unchecked | Set up `logrotate` |
| Month 1 | Message bus slow | Too many files in inbox | Archive old messages, shard by date |

### 4.2 Emergency Procedures

**Emergency: DGX job runaway spend**
```bash
# Atlas kills all Moses jobs
ssh dgx 'scancel -u moses'
# Renaissance locks envelope
python3 -c "from human_checkpoint import HITLGuard; ... # lock"
# Walker alerts Alex immediately
```

**Emergency: Safety-critical bug in deployed policy**
```bash
# Titan triggers emergency stop
# Moses halts all physical deploys
# Walker creates EMERGENCY_STOP checkpoint
# Alex must APPROVE before any resume
```

**Emergency: Message bus total failure**
```bash
# All agents fallback to local outbox
# Walker polls local outboxes every 5 min
# If >30 min, alert Alex via Telegram direct message
```

---

## 5. Post-Activation Checklist

- [ ] All 5 files committed to git
- [ ] Alex has tested at least one Tier 3 gate end-to-end
- [ ] Renaissance budget envelope confirmed
- [ ] Atlas CI pipeline green on `main`
- [ ] Titan simulation runs on default scenario
- [ ] Moses can submit design and receive report
- [ ] Walker can view all agent statuses in Mission Control
- [ ] `.learnings/ERRORS.md` and `.learnings/LEARNINGS.md` initialized
- [ ] Rollback plan documented (disable crons, revert to manual)

---

*Activation Sequence v1.0 — Boss Industries — 2026-06-08*
