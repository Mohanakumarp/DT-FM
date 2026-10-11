const app = document.querySelector("#app");
const dialog = document.querySelector("#dialog");
const state = {
  auth: null,
  catalog: null,
  clusters: [],
  cluster: null,
  agents: [],
  jobs: [],
  screen: "clusters",
  tab: "overview",
  job: null,
  events: [],
  draft: null,
  refreshing: false,
  register: false,
};
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );
const symbols = {
  chip: '<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m-3 6h3m14-6h3m-3 6h3"/>',
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  nodes:
    '<rect x="8" y="2" width="8" height="6" rx="1"/><rect x="2" y="16" width="8" height="6" rx="1"/><rect x="14" y="16" width="8" height="6" rx="1"/><path d="M12 8v4m-6 4v-4h12v4"/>',
  activity: '<path d="M2 12h4l3-8 6 16 3-8h4"/>',
  people:
    '<circle cx="9" cy="7" r="3"/><path d="M3 21v-3a6 6 0 0 1 12 0v3m1-17a3 3 0 0 1 0 6m3 11v-3a6 6 0 0 0-3-5"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>',
  back: '<path d="M19 12H5m5-5-5 5 5 5"/>',
  logout: '<path d="M9 4H4v16h5m5-13 5 5-5 5M9 12h10"/>',
  play: '<path d="m7 4 14 8-14 8z"/>',
  stop: '<rect x="5" y="5" width="14" height="14" rx="2"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>',
  server:
    '<rect x="3" y="3" width="18" height="7" rx="2"/><rect x="3" y="14" width="18" height="7" rx="2"/><path d="M7 6.5h.01M7 17.5h.01m4-11h6m-6 11h6"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
};
const icon = (name) =>
  `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${symbols[name] || symbols.grid}</svg>`;
const brand = () =>
  `<div class="brand"><div class="brand-mark">${icon("chip")}</div>DT-FM<span>CONSOLE</span></div>`;
const badge = (value, label = value) =>
  `<span class="badge ${["approved", "completed", "available", "owner", "succeeded", "online"].includes(value) ? "green" : ["running", "preparing", "pending", "ready", "cancelling"].includes(value) ? "amber" : ["failed", "revoked", "rejected"].includes(value) ? "red" : ""}">${esc(label)}</span>`;
const date = (value) =>
  new Date(value * 1000).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
const memory = (bytes) =>
  bytes == null ? "Not reported" : `${(bytes / 2 ** 30).toFixed(1)} GB`;
const isOwner = () => state.cluster?.owner_id === state.auth?.user.id;
const empty = (title, copy, action = "") =>
  `<div class="empty">${icon("nodes")}<h3>${esc(title)}</h3><p>${esc(copy)}</p>${action}</div>`;
const field = (name, label, value, options = {}) =>
  `<label class="field ${options.full ? "full" : ""}"><span>${esc(label)}</span>${options.select ? `<select name="${esc(name)}">${options.select.map(([key, text]) => `<option value="${esc(key)}" ${String(value) === String(key) ? "selected" : ""}>${esc(text)}</option>`).join("")}</select>` : `<input name="${esc(name)}" type="${options.type || "text"}" value="${esc(value)}" ${options.min != null ? `min="${options.min}"` : ""} ${options.max != null ? `max="${options.max}"` : ""} ${options.step ? `step="${options.step}"` : ""} ${options.required === false ? "" : "required"} ${options.autocomplete ? `autocomplete="${options.autocomplete}"` : ""}>`}${options.help ? `<small>${esc(options.help)}</small>` : ""}</label>`;

async function api(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      "X-CSRF-Token": state.auth?.csrf || "",
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const data = await response.json();
  if (!response.ok) {
    if (response.status === 401 && !path.includes("/auth/")) {
      state.auth = null;
      dialog.close();
      render();
    }
    throw new Error(data.error || `Request failed (${response.status})`);
  }
  return data;
}

function toast(message, error = false) {
  const target = document.querySelector("#toast");
  target.textContent = message;
  target.className = error ? "error" : "";
  target.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(
    () => {
      target.hidden = true;
    },
    error ? 7000 : 4000,
  );
}

function modal(title, content) {
  dialog.innerHTML = `<div class="dialog-head"><h2 id="dialog-title">${esc(title)}</h2><button class="quiet small" data-action="close-dialog" aria-label="Close dialog">${icon("close")}</button></div><div class="dialog-body">${content}<div class="dialog-error" role="alert"></div></div>`;
  if (!dialog.open) dialog.showModal();
}

async function refreshClusters() {
  state.clusters = (await api("/api/clusters")).clusters;
}

async function refreshCluster() {
  const id = state.cluster.id;
  const [cluster, agents, jobs] = await Promise.all([
    api(`/api/clusters/${id}`),
    api(`/api/clusters/${id}/agents`),
    api(`/api/clusters/${id}/jobs`),
  ]);
  if (state.cluster?.id !== id) return;
  state.cluster = cluster;
  state.agents = agents.agents;
  state.jobs = jobs.jobs;
  if (!state.draft) {
    const available = eligibleDevices();
    state.draft = {
      config: { ...state.catalog.defaults },
      devices: available.length ? [available[0].key] : [],
    };
  }
}

async function openCluster(id, tab = "overview") {
  state.cluster = { id };
  state.tab = tab;
  state.screen = "cluster";
  state.draft = null;
  app.innerHTML = '<p class="loading">Loading cluster…</p>';
  try {
    await refreshCluster();
    render();
  } catch (error) {
    state.cluster = null;
    state.screen = "clusters";
    render();
    throw error;
  }
}

function allDevices() {
  return state.agents.flatMap((agent) =>
    agent.capabilities.devices.map((device) => ({
      agent,
      device,
      key: `${agent.id}:${device.id}`,
      supported: ["cpu", "cuda"].includes(device.backend),
      reservation: agent.reservations?.find(
        (item) => item.device_id === device.id,
      ),
      available: Boolean(
        agent.online &&
          agent.enabled &&
          agent.capabilities.runtime_ready &&
          agent.shared_devices.includes(device.id) &&
          !agent.reservations?.some((item) => item.device_id === device.id),
      ),
    })),
  );
}
function eligibleDevices() {
  return allDevices().filter((item) => item.available && item.supported);
}

function authPage() {
  return `<main class="auth-shell"><section class="auth-story">${brand()}<h1>Bring your<br>compute <span>together.</span></h1><p>A shared workspace for distributed training. Connect your devices, build a cluster, and turn individual contributions into a training run.</p><div class="grid-art" aria-hidden="true">${"<span></span>".repeat(20)}</div><footer><span>OWNER-CONTROLLED TRAINING</span><span>CONTRIBUTOR-CONTROLLED COMPUTE</span></footer></section><section class="auth-form-wrap"><form class="auth-form" data-form="auth"><div class="eyebrow">CLUSTER CONSOLE</div><br><h2>${state.register ? "Create your account" : "Welcome back"}</h2><p>${state.register ? "Join a cluster or create a workspace for your team." : "Sign in to manage your clusters and contributions."}</p>${state.register ? field("name", "Your name", "", { autocomplete: "name" }) : ""}${field("email", "Email address", "", { type: "email", autocomplete: "username" })}${field("password", "Password", "", { type: "password", autocomplete: state.register ? "new-password" : "current-password", help: state.register ? "At least 12 characters." : "" })}<button class="primary" type="submit">${state.register ? "Create account" : "Sign in"} ${icon("arrow")}</button><div class="auth-error" role="alert"></div><div class="auth-switch">${state.register ? "Already have an account?" : "New to DT-FM?"} <button type="button" class="quiet" data-action="toggle-auth">${state.register ? "Sign in" : "Create an account"}</button></div></form></section></main>`;
}

function stats(items) {
  return `<div class="stats">${items.map((item) => `<div class="stat"><div class="stat-label">${esc(item.label)}${icon(item.icon)}</div><div class="stat-value">${esc(item.value)}${item.unit ? `<small>${esc(item.unit)}</small>` : ""}</div><div class="stat-foot ${item.accent ? "accent" : ""}">${esc(item.foot)}</div></div>`).join("")}</div>`;
}

function clusterCards() {
  return state.clusters.length
    ? `<div class="cards">${state.clusters.map((cluster) => `<article class="cluster-card"><div class="card-top"><div class="cluster-icon">${icon("nodes")}</div><div><h3>${esc(cluster.name)}</h3><p>${esc(cluster.description || "A workspace for shared compute and distributed training.")}</p></div></div><div class="card-details"><span>${icon("people")} ${cluster.member_count} member${cluster.member_count === 1 ? "" : "s"}</span><span>Owner · ${esc(cluster.owner_name)}</span></div><div class="card-footer">${badge(cluster.membership || "open", cluster.role === "owner" ? "Your cluster" : cluster.membership === "approved" ? "Contributor" : cluster.membership || "Open to requests")}${cluster.membership === "approved" ? `<button class="quiet small" data-action="open-cluster" data-id="${cluster.id}">Open cluster ${icon("arrow")}</button>` : cluster.membership === "pending" ? '<span class="muted">Awaiting approval</span>' : `<button class="small" data-action="join" data-id="${cluster.id}">Request to join</button>`}</div></article>`).join("")}</div>`
    : empty(
        "Your first cluster starts here",
        "Create a cluster, invite contributors, and bring your devices online.",
        '<button class="primary" data-action="create-cluster">Create cluster</button>',
      );
}

function clustersPage() {
  const joined = state.clusters.filter((c) => c.membership === "approved");
  return `<div class="page-head"><div><div class="eyebrow">YOUR WORKSPACE</div><h1>Clusters</h1><p>Bring devices together. Keep training in sync.</p></div><button class="primary" data-action="create-cluster">${icon("plus")} Create cluster</button></div>${stats(
    [
      {
        label: "Your clusters",
        value: joined.length,
        icon: "nodes",
        foot: "Approved memberships",
        accent: true,
      },
      {
        label: "Owned by you",
        value: joined.filter((c) => c.role === "owner").length,
        icon: "grid",
        foot: "You configure the training",
      },
      {
        label: "Join requests",
        value: state.clusters.filter((c) => c.membership === "pending").length,
        icon: "people",
        foot: "Waiting for owner approval",
      },
      {
        label: "Discoverable clusters",
        value: state.clusters.length,
        icon: "server",
        foot: "On this control server",
      },
    ],
  )}<div class="section-head"><h2>Cluster directory</h2><small>${state.clusters.length} workspace${state.clusters.length === 1 ? "" : "s"}</small></div>${clusterCards()}<div class="section-head"><h2>From contribution to training</h2></div><div class="panel"><div class="panel-body"><ol class="help-list"><li><span class="step">1</span><div><strong>Join a workspace</strong><br>Create a cluster or request membership. The owner approves contributors.</div></li><li><span class="step">2</span><div><strong>Connect a computer</strong><br>Pair the local worker agent and choose the devices you want to share.</div></li><li><span class="step">3</span><div><strong>Train together</strong><br>The owner chooses a model, dataset, and topology. Workers prepare and start together.</div></li></ol></div></div>`;
}

function runRows(limit = 100) {
  return state.jobs
    .slice(0, limit)
    .map(
      (job) =>
        `<div class="row run-card" role="button" tabindex="0" data-action="open-job" data-id="${job.id}"><div class="cluster-icon">${icon("activity")}</div><div><div class="row-title">${esc(state.catalog.models[job.spec.config.model]?.label || job.spec.config.model)}</div><div class="row-sub">${job.spec.world_size} ranks · ${job.spec.config.pipeline_size} stages × ${job.spec.config.replicas} replicas · ${date(job.created)}</div></div><div class="row-end">${badge(job.status)}${icon("arrow")}</div></div>`,
    )
    .join("");
}

function overviewPage() {
  const devices = allDevices();
  const online = state.agents.filter((a) => a.online).length;
  return `${stats([
    {
      label: "Connected computers",
      value: online,
      unit: `/ ${state.agents.length}`,
      icon: "server",
      foot: "Heartbeats update automatically",
      accent: true,
    },
    {
      label: "Available devices",
      value: eligibleDevices().length,
      icon: "chip",
      foot: `${devices.filter((d) => d.device.backend === "cuda").length} NVIDIA GPUs registered`,
    },
    {
      label: "Cluster members",
      value: state.cluster.members.filter((m) => m.status === "approved")
        .length,
      icon: "people",
      foot: `${state.cluster.members.filter((m) => m.status === "pending").length} requests awaiting approval`,
    },
    {
      label: "Training runs",
      value: state.jobs.length,
      icon: "activity",
      foot: `${state.jobs.filter((j) => ["preparing", "running"].includes(j.status)).length} active · ${state.jobs.filter((j) => j.status === "completed").length} completed`,
    },
  ])}<div class="section-head"><h2>Cluster activity</h2>${isOwner() ? '<button class="small" data-action="tab" data-tab="training">Configure a run</button>' : ""}</div><div class="two-column"><section class="panel"><div class="panel-head"><h2>Recent training</h2><span class="muted">${state.jobs.length} runs</span></div>${state.jobs.length ? runRows(5) : empty("Ready when your devices are", "Connect at least one supported device to start your first training run.")}</section><section class="panel"><div class="panel-head"><h2>Getting connected</h2></div><div class="panel-body"><ol class="help-list"><li><span class="step">1</span><div><strong>Pair your computer</strong><br>Install the training dependencies, then pair your local worker.</div></li><li><span class="step">2</span><div><strong>Choose your contribution</strong><br>Share a CPU or NVIDIA GPU. You can pause sharing at any time.</div></li><li><span class="step">3</span><div><strong>Check connectivity</strong><br>Across computers, use reachable private addresses and a working Gloo network.</div></li></ol><br><button data-action="pair">${icon("plus")} Connect computer</button></div></section></div>`;
}

function devicesPage() {
  const devices = allDevices();
  return `<div class="section-head"><div><h2>Contributed devices</h2><small>Each contributor controls the devices their worker is allowed to use.</small></div><button class="primary" data-action="pair">${icon("plus")} Connect computer</button></div><section class="panel">${devices.length ? `<div class="table-wrap"><table><thead><tr><th>Device</th><th>Computer / contributor</th><th>Memory</th><th>Runtime</th><th>Contribution</th></tr></thead><tbody>${devices.map(({ agent, device, supported, available, reservation }) => `<tr><td class="device-cell"><div class="device-name">${icon("chip")}<div><div class="row-title">${esc(device.name)}</div><div class="row-sub">${esc(device.backend.toUpperCase())} · ${esc(device.id)}</div></div></div></td><td><div>${esc(agent.name)}</div><div class="row-sub">${esc(agent.contributor)} · ${esc(agent.address)}</div></td><td class="mono">${memory(device.memory_bytes)}</td><td>${badge(agent.online ? "online" : "offline")}${!agent.capabilities.runtime_ready ? `<div class="row-sub">${esc(agent.capabilities.runtime_error)}</div>` : !supported ? '<div class="row-sub">Training adapter not available</div>' : ""}</td><td>${badge(reservation ? reservation.state : available && supported ? "available" : "paused", reservation ? (reservation.state === "running" ? "Training" : "Reserved") : available && supported ? "Shared" : !agent.enabled ? "Paused" : !agent.shared_devices.includes(device.id) ? "Not shared" : "Unavailable")}${agent.user_id === state.auth.user.id ? `<button class="quiet small" data-action="sharing" data-id="${agent.id}">${agent.enabled ? "Pause computer" : "Resume computer"}</button>` : ""}</td></tr>`).join("")}</tbody></table></div>` : empty("No computers connected yet", "Pair a worker on your computer to discover and contribute its devices.")}</section><br><div class="note">This first version runs QA and smoke jobs on CPU and NVIDIA CUDA, using Gloo and FP32. Other detected accelerators remain visible so their support status is clear.</div>`;
}

function trainingForm() {
  const draft = state.draft;
  draft.devices = draft.devices.filter((key) =>
    eligibleDevices().some((item) => item.key === key),
  );
  const c = draft.config;
  const smoke = c.workflow === "smoke";
  const models = Object.entries(state.catalog.models)
    .filter(([key]) => smoke === (key === "tiny-random-bert"))
    .map(([key, value]) => [key, value.label]);
  const available = eligibleDevices();
  return `<form data-form="training"><div class="two-column"><section class="panel"><div class="panel-head"><h2>Training configuration</h2><span class="badge">FP32 · Gloo</span></div><div class="panel-body"><div class="form-grid">${field(
    "workflow",
    "Training workflow",
    c.workflow,
    {
      select: [
        ["smoke", "Offline smoke · no downloads"],
        ["qa", "BERT question answering"],
      ],
    },
  )}${field("model", "Model", c.model, { select: models })}${field("dataset", "Dataset", c.dataset, { full: true, help: smoke ? "Synthetic spans generated locally. No dataset download." : "Hugging Face ID, e.g. rajpurkar/squad. Requires SQuAD fields and train / validation splits." })}</div><h3 class="form-section">Parallelism</h3><div class="form-grid">${field("pipeline_size", "Stages per pipeline", c.pipeline_size, { type: "number", min: 1, max: state.catalog.models[c.model].units, help: "Each stage runs one partition of the model." })}${field("replicas", "Pipeline replicas", c.replicas, { type: "number", min: 1, max: 16, help: "Data parallelism across complete pipelines." })}</div><div class="topology"><span>Stages × replicas = required ranks</span><strong id="rank-total">${c.pipeline_size} × ${c.replicas} = ${c.pipeline_size * c.replicas}</strong></div><h3 class="form-section">Training parameters</h3><div class="form-grid">${field("batch_size", "Batch size per replica", c.batch_size, { type: "number", min: 1, max: 1024 })}${field("micro_batch_size", "Microbatch size", c.micro_batch_size, { type: "number", min: 1, max: 1024 })}${field("epochs", "Epochs", c.epochs, { type: "number", min: 1, max: 10000 })}${field("max_steps", "Maximum optimizer steps", c.max_steps, { type: "number", min: 0, max: 1000000, help: "0 runs all configured epochs." })}${field("learning_rate", "Learning rate", c.learning_rate, { type: "number", min: 0.000000001, max: 1, step: "any" })}${field("checkpoint_every", "Checkpoint interval (steps)", c.checkpoint_every, { type: "number", min: 0, max: 100000, help: "0 saves at the end. Files stay on each worker." })}</div><details class="advanced"><summary>Advanced settings</summary><div class="form-grid">${field("max_length", "Sequence length", c.max_length, { type: "number", min: 8, max: 512 })}${field("doc_stride", "Document stride", c.doc_stride, { type: "number", min: 0, max: 511 })}${field("train_examples", "Training examples", c.train_examples, { type: "number", min: 1, max: 1000000 })}${field("validation_examples", "Validation examples", c.validation_examples, { type: "number", min: 1, max: 100000 })}${field("model_revision", "Model revision", c.model_revision, { help: "Use a commit hash to pin the model." })}${field("dataset_revision", "Dataset revision", c.dataset_revision, { help: "Use a commit hash to pin the dataset." })}${field("port", "Coordinator TCP port", c.port, { type: "number", min: 1024, max: 65535 })}${field("timeout_seconds", "Preparation / communication timeout", c.timeout_seconds, { type: "number", min: 10, max: 1800 })}</div></details></div></section><section class="panel"><div class="panel-head"><h2>Participating devices</h2><span class="muted">${available.length} eligible</span></div><div class="panel-body"><p class="muted row-sub">Select one device per rank. Selection order determines rank order; rank 0 hosts the coordinator.</p><div id="device-selections">${available.length ? available.map(({ agent, device, key }) => `<label class="device-choice"><input type="checkbox" name="device" value="${esc(key)}" ${draft.devices.includes(key) ? "checked" : ""}><span>${esc(agent.name)}<br><span class="muted">${esc(device.name)} · ${esc(device.backend.toUpperCase())}</span></span><span class="badge">${draft.devices.includes(key) ? `Rank ${draft.devices.indexOf(key)}` : "—"}</span></label>`).join("") : '<div class="empty"><h3>No eligible devices</h3><p>Connect a worker and share a supported device.</p></div>'}</div><div class="topology"><span>Selected / required</span><strong id="selection-total">${draft.devices.length} / ${c.pipeline_size * c.replicas}</strong></div><div class="note warning">Workers prepare the model and dataset before training. All ranks must be ready; a failed rank stops the whole job.</div><div class="form-actions"><button class="primary" type="submit" ${draft.devices.length !== c.pipeline_size * c.replicas ? "disabled" : ""}>Review assignments ${icon("arrow")}</button></div></div></section></div></form>`;
}

function trainingPage() {
  return `${isOwner() ? trainingForm() : '<div class="note">The cluster owner configures and starts training. You can follow every run here.</div>'}<div class="section-head"><h2>Training history</h2><small>Configuration snapshots are saved with each run</small></div><section class="panel">${state.jobs.length ? runRows() : empty("No training runs yet", "The owner can configure a run once supported devices are available.")}</section>`;
}

function membersPage() {
  return `<div class="section-head"><div><h2>Members & requests</h2><small>Only approved members can connect workers or view training details.</small></div></div><section class="panel"><div class="table-wrap"><table><thead><tr><th>Member</th><th>Role</th><th>Membership</th><th>Actions</th></tr></thead><tbody>${state.cluster.members.map((member) => `<tr><td><div class="device-name"><span class="avatar">${esc(member.name.slice(0, 2).toUpperCase())}</span><span>${esc(member.name)}${member.id === state.auth.user.id ? " (you)" : ""}</span></div></td><td>${badge(member.role)}</td><td>${badge(member.status)}</td><td>${isOwner() && member.role !== "owner" ? (member.status === "pending" ? `<button class="small primary" data-action="membership" data-id="${member.id}" data-decision="approve">Approve</button> <button class="small quiet" data-action="membership" data-id="${member.id}" data-decision="reject">Reject</button>` : member.status === "approved" ? `<button class="small quiet danger" data-action="membership" data-id="${member.id}" data-decision="revoke">Revoke access</button>` : '<span class="muted">A new request is required</span>') : '<span class="muted">—</span>'}</td></tr>`).join("")}</tbody></table></div></section><br><div class="note">Revoking membership disconnects that contributor's workers and stops jobs using their devices.</div>`;
}

function clusterPage() {
  return `<div class="page-head"><div><div class="eyebrow">CLUSTER WORKSPACE</div><h1>${esc(state.cluster.name)}</h1><p>${esc(state.cluster.description || "Shared compute. Coordinated training.")}</p></div><div class="head-actions">${badge(isOwner() ? "owner" : "approved", isOwner() ? "Cluster owner" : "Contributor")}<button data-action="pair">${icon("plus")} Connect computer</button></div></div><nav class="tabs" aria-label="Cluster sections">${[
    ["overview", "Overview"],
    ["devices", "Devices"],
    ["training", "Training"],
    ["members", "Members"],
  ]
    .map(
      ([tab, label]) =>
        `<button data-action="tab" data-tab="${tab}" class="${state.tab === tab ? "active" : ""}">${label}</button>`,
    )
    .join(
      "",
    )}</nav>${{ overview: overviewPage, devices: devicesPage, training: trainingPage, members: membersPage }[state.tab]()}`;
}

function lossPoints() {
  const values = new Map();
  for (const event of state.events) {
    if (event.rank !== state.job.spec.world_size - 1) continue;
    for (const line of event.message.split("\n")) {
      try {
        const item = JSON.parse(line);
        if (Number.isFinite(item.train_loss) && Number.isInteger(item.step))
          values.set(item.step, item.train_loss);
      } catch {
        /* Human-readable diagnostics remain in the log. */
      }
    }
  }
  return [...values].sort((a, b) => a[0] - b[0]);
}

function lossChart() {
  const losses = lossPoints();
  if (!losses.length)
    return '<p class="row-sub">Loss will appear after the first optimizer step.</p>';
  const min = Math.min(...losses.map((p) => p[1]));
  const max = Math.max(...losses.map((p) => p[1]));
  const points = losses
    .map(
      ([, loss], i) =>
        `${10 + (i * 480) / Math.max(1, losses.length - 1)},${75 - ((loss - min) / Math.max(0.001, max - min)) * 55}`,
    )
    .join(" ");
  return `<svg viewBox="0 0 500 95" width="100%" height="110" role="img" aria-label="Training loss across ${losses.length} optimizer steps"><polyline points="${points}" fill="none" stroke="#74e7b8" stroke-width="2"/>${losses.length === 1 ? '<circle cx="10" cy="75" r="3" fill="#74e7b8"/>' : ""}</svg><div class="chart-labels"><span>Step ${losses[0][0]}</span><span>Loss ${losses.at(-1)[1].toFixed(4)} · step ${losses.at(-1)[0]}</span></div>`;
}

function jobPage() {
  const job = state.job;
  const config = job.spec.config;
  const final = job.assignments.at(-1)?.metrics || {};
  const steps = Math.min(
    ...job.assignments.map(
      (a) => a.metrics.completed_steps || a.metrics.step || 0,
    ),
  );
  return `<div class="page-head"><div><button class="quiet small" data-action="back-training">${icon("back")} Training</button><h1>${esc(state.catalog.models[config.model]?.label || config.model)}</h1><p>${esc(config.dataset)} · Started ${date(job.created)} · <span class="mono">${job.id.slice(0, 8)}</span></p></div><div class="head-actions">${badge(job.status)}${isOwner() && job.status === "draft" ? `<button class="primary" data-action="start-job" data-id="${job.id}">${icon("play")} Start training</button>` : ""}${isOwner() && ["draft", "preparing", "running"].includes(job.status) ? `<button class="danger" data-action="cancel-job" data-id="${job.id}">${icon("stop")} Stop run</button>` : ""}</div></div>${job.error ? `<div class="note warning">${esc(job.error)}</div><br>` : ""}<div class="two-column"><section class="panel"><div class="panel-head"><h2>Training progress</h2><span class="badge">FP32 · Gloo</span></div><div class="panel-body"><div class="job-metrics"><div><span>COMPLETED STEPS</span><strong>${steps}</strong></div><div><span>LATEST LOSS</span><strong>${Number.isFinite(final.train_loss) ? final.train_loss.toFixed(4) : "—"}</strong></div><div><span>FEATURES / SECOND</span><strong>${Number.isFinite(final.features_per_second) ? final.features_per_second.toFixed(1) : "—"}</strong></div></div>${lossChart()}</div></section><section class="panel"><div class="panel-head"><h2>Configuration snapshot</h2></div><div class="panel-body"><div class="row-sub">${config.pipeline_size} stages × ${config.replicas} replicas = ${job.spec.world_size} ranks<br>Batch ${config.batch_size} · microbatch ${config.micro_batch_size}<br>${config.epochs} epochs · step cap ${config.max_steps || "none"}<br>Learning rate ${config.learning_rate}<br>Model revision: ${esc(config.model_revision)}<br>Dataset revision: ${esc(config.dataset_revision)}<br>Checkpoints: every ${config.checkpoint_every || "final"} step${config.checkpoint_every === 1 ? "" : "s"}</div></div></section></div><div class="section-head"><h2>Rank assignments</h2><small>Fixed for this run</small></div><section class="panel"><div class="table-wrap"><table><thead><tr><th>Rank</th><th>Computer / device</th><th>Pipeline</th><th>Status</th><th>Current phase</th></tr></thead><tbody>${job.assignments
    .map((a) => {
      const assignment = job.spec.assignments[a.rank];
      return `<tr><td class="mono">${a.rank}</td><td>${esc(assignment.computer)}<div class="row-sub">${esc(assignment.device.name)} · ${esc(assignment.device.backend.toUpperCase())}</div></td><td>Replica ${assignment.replica} · stage ${assignment.pipeline_stage}</td><td>${badge(a.state)}</td><td class="muted">${esc(a.metrics.phase || "Awaiting worker")}</td></tr>`;
    })
    .join(
      "",
    )}</tbody></table></div></section><div class="section-head"><h2>Worker logs</h2><small>Last ${Math.min(160, state.events.length)} events · full logs remain on workers</small></div><section class="panel"><pre class="log" id="run-log">${esc(
    state.events
      .slice(-160)
      .map(
        (event) =>
          `[${new Date(event.timestamp * 1000).toLocaleTimeString()}${event.rank == null ? " · coordinator" : ` · rank ${event.rank}`}] ${event.message}`,
      )
      .join("\n") || "Waiting for worker events…",
  )}</pre></section><br><p class="row-sub">Checkpoint and model artifacts are saved in each worker's configured output directory. Automatic job recovery and artifact downloads are not included in this version.</p>`;
}

function render() {
  if (!state.auth) {
    app.innerHTML = authPage();
    return;
  }
  const sections = [
    ["overview", "Overview", "grid"],
    ["devices", "Devices", "chip"],
    ["training", "Training", "activity"],
    ["members", "Members", "people"],
  ];
  const name = state.auth.user.name;
  app.innerHTML = `<div class="shell"><aside class="sidebar">${brand()}<div><div class="nav-label">WORKSPACE</div><nav class="nav" aria-label="Main navigation"><button data-action="clusters" class="${state.screen === "clusters" ? "active" : ""}">${icon("nodes")} All clusters</button>${state.cluster ? sections.map(([tab, label, image]) => `<button data-action="tab" data-tab="${tab}" class="${state.screen !== "clusters" && state.tab === tab ? "active" : ""}">${icon(image)} ${label}</button>`).join("") : ""}</nav></div><div class="sidebar-bottom"><div class="sidebar-note"><span class="dot"></span>Connected to DT-FM<br>Local agents execute training.<br>You control your contribution.</div><div class="user"><span class="avatar">${esc(name.slice(0, 2).toUpperCase())}</span><div><div class="user-name">${esc(name)}</div><div class="user-meta">Cluster workspace</div></div><button class="quiet small" data-action="logout" aria-label="Sign out">${icon("logout")}</button></div></div></aside><main class="main"><header class="topbar"><div class="breadcrumb"><button class="quiet" data-action="clusters">Workspace</button><span>/</span><span>${state.screen === "clusters" ? "Clusters" : esc(state.cluster.name)}</span>${state.screen === "job" ? "<span>/</span><span>Training run</span>" : ""}</div><div class="connection"><span class="dot"></span>Control server connected<button class="quiet small" data-action="logout" aria-label="Sign out">${icon("logout")}</button></div></header><div class="content">${state.screen === "clusters" ? clustersPage() : state.screen === "job" ? jobPage() : clusterPage()}</div></main></div>`;
}

async function fetchJob(id, reset = false) {
  if (reset) state.events = [];
  const cursor = state.events.at(-1)?.id || 0;
  const job = await api(`/api/jobs/${id}?after=${cursor}`);
  state.events.push(...job.events);
  state.events = state.events.slice(-2000);
  state.job = job;
}

function updateDraft(form) {
  const data = new FormData(form);
  for (const key of Object.keys(state.catalog.defaults)) {
    if (data.has(key))
      state.draft.config[key] =
        typeof state.catalog.defaults[key] === "number"
          ? Number(data.get(key))
          : data.get(key);
  }
  const selected = data.getAll("device");
  state.draft.devices = [
    ...state.draft.devices.filter((key) => selected.includes(key)),
    ...selected.filter((key) => !state.draft.devices.includes(key)),
  ];
}

async function handleAction(target) {
  const action = target.dataset.action;
  const id = target.dataset.id;
  if (action === "toggle-auth") {
    state.register = !state.register;
    render();
  }
  if (action === "close-dialog") dialog.close();
  if (action === "logout") {
    await api("/api/auth/logout", {});
    Object.assign(state, {
      auth: null,
      cluster: null,
      jobs: [],
      agents: [],
      job: null,
      events: [],
      draft: null,
      screen: "clusters",
    });
    dialog.close();
    render();
  }
  if (action === "clusters") {
    state.screen = "clusters";
    await refreshClusters();
    render();
  }
  if (action === "open-cluster") await openCluster(id);
  if (action === "tab" || action === "back-training") {
    state.screen = "cluster";
    state.tab = action === "back-training" ? "training" : target.dataset.tab;
    await refreshCluster();
    render();
  }
  if (action === "create-cluster")
    modal(
      "Create a cluster",
      `<p>A workspace for your contributors, devices, and training runs. You will be its owner.</p><form data-form="create-cluster"><div class="form-grid">${field("name", "Cluster name", "", { full: true })}<label class="field full"><span>Description</span><textarea name="description" rows="3" maxlength="500" placeholder="What will your team train?"></textarea></label></div><div class="form-actions"><button class="primary" type="submit">Create cluster ${icon("arrow")}</button></div></form>`,
    );
  if (action === "join") {
    await api(`/api/clusters/${id}/join`, {});
    await refreshClusters();
    render();
    toast("Request sent. The cluster owner can approve your membership.");
  }
  if (action === "membership") {
    await api(`/api/clusters/${state.cluster.id}/members/${id}`, {
      decision: target.dataset.decision,
    });
    await refreshCluster();
    render();
    toast("Membership updated.");
  }
  if (action === "sharing") {
    const agent = state.agents.find((a) => a.id === id);
    await api(`/api/clusters/${state.cluster.id}/agents/${id}/sharing`, {
      enabled: !agent.enabled,
      shared_devices: agent.shared_devices,
    });
    await refreshCluster();
    render();
    toast(
      agent.enabled
        ? "Sharing paused. Active jobs using this computer will stop."
        : "Computer sharing resumed.",
    );
  }
  if (action === "pair") {
    const pair = await api(`/api/clusters/${state.cluster.id}/pairing`, {});
    modal(
      "Connect your computer",
      `<p>Run these commands from your DT-FM checkout using the Python environment that has your training dependencies installed. These commands work in Bash and PowerShell.</p><div class="eyebrow">YOUR ONE-TIME PAIRING CODE · EXPIRES IN 10 MINUTES</div><pre class="pair-code">${esc(pair.code)}</pre><div class="eyebrow">1 · REGISTER YOUR COMPUTER</div><pre>python -m cluster_control.worker pair --server ${esc(location.origin)} --address &lt;private-IP&gt; --devices cpu</pre><p>Paste the code when prompted. For an NVIDIA GPU, replace <code>cpu</code> with <code>cuda:0</code>. Use <code>python -m cluster_control.worker inspect</code> to see your devices. Across computers, use a reachable private address such as your Tailscale IPv4.</p><div class="eyebrow">2 · START CONTRIBUTING</div><pre>python -m cluster_control.worker run</pre><p>Keep the agent running. Ctrl+C stops its local training. You can also pause sharing from the Devices page.</p><button class="primary" data-action="close-dialog">Done</button>`,
    );
  }
  if (action === "open-job") {
    await fetchJob(id, true);
    state.screen = "job";
    state.tab = "training";
    render();
  }
  if (action === "start-job") {
    await api(`/api/jobs/${id}/start`, {});
    await fetchJob(id);
    render();
    toast("Workers are preparing the job.");
  }
  if (action === "cancel-job") {
    await api(`/api/jobs/${id}/cancel`, {});
    await fetchJob(id);
    render();
    toast("Stop requested. Waiting for the workers to acknowledge.");
  }
  if (action === "save-job" || action === "launch-job") {
    const created = await api(
      `/api/clusters/${state.cluster.id}/jobs`,
      state.draft,
    );
    if (action === "launch-job") {
      try {
        await api(`/api/jobs/${created.id}/start`, {});
      } catch (error) {
        toast(`Draft saved. ${error.message}`, true);
      }
    }
    dialog.close();
    await fetchJob(created.id, true);
    state.screen = "job";
    state.tab = "training";
    render();
  }
}

document.addEventListener("click", async (event) => {
  const target = event.target.closest("[data-action]");
  if (!target || target.disabled) return;
  try {
    target.disabled = true;
    await handleAction(target);
  } catch (error) {
    toast(error.message, true);
  } finally {
    if (target.isConnected) target.disabled = false;
  }
});

document.addEventListener("keydown", (event) => {
  const target = event.target.closest('[role="button"][data-action]');
  if (target && ["Enter", " "].includes(event.key)) {
    event.preventDefault();
    target.click();
  }
});

document.addEventListener("submit", async (event) => {
  const form = event.target.closest("[data-form]");
  if (!form) return;
  event.preventDefault();
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  try {
    if (form.getAttribute("data-form") === "auth") {
      const auth = await api(
        `/api/auth/${state.register ? "register" : "login"}`,
        Object.fromEntries(new FormData(form)),
      );
      state.auth = auth;
      state.catalog = await api("/api/catalog");
      await refreshClusters();
      render();
    }
    if (form.getAttribute("data-form") === "create-cluster") {
      const cluster = await api(
        "/api/clusters",
        Object.fromEntries(new FormData(form)),
      );
      dialog.close();
      await refreshClusters();
      await openCluster(cluster.id);
      toast("Cluster created. You can now connect computers.");
    }
    if (form.getAttribute("data-form") === "training") {
      updateDraft(form);
      const plan = await api(
        `/api/clusters/${state.cluster.id}/preview`,
        state.draft,
      );
      modal(
        "Review training assignments",
        `<p>${plan.config.pipeline_size} stages × ${plan.config.replicas} replicas = ${plan.world_size} ranks. The owner settings below will be saved as a fixed configuration snapshot.</p><div class="table-wrap"><table><thead><tr><th>Rank</th><th>Computer</th><th>Device</th><th>Placement</th></tr></thead><tbody>${plan.assignments.map((a) => `<tr><td>${a.rank}</td><td>${esc(a.computer)}</td><td>${esc(a.device.backend.toUpperCase())}</td><td>Replica ${a.replica}, stage ${a.pipeline_stage}</td></tr>`).join("")}</tbody></table></div><br><div class="note warning">${plan.warnings.map(esc).join("<br>")}</div><div class="form-actions"><button data-action="save-job">Save draft</button><button class="primary" data-action="launch-job">${icon("play")} Start training</button></div>`,
      );
    }
  } catch (error) {
    const message =
      form.querySelector(".auth-error") ||
      (dialog.open ? dialog.querySelector(".dialog-error") : null);
    if (message) message.textContent = error.message;
    else toast(error.message, true);
  } finally {
    if (button.isConnected) button.disabled = false;
  }
});

document.addEventListener("input", (event) => {
  const form = event.target.closest('[data-form="training"]');
  if (!form) return;
  updateDraft(form);
  const c = state.draft.config;
  const total = c.pipeline_size * c.replicas;
  form.elements.namedItem("pipeline_size").max =
    state.catalog.models[c.model].units;
  form.querySelector("#rank-total").textContent =
    `${c.pipeline_size} × ${c.replicas} = ${total}`;
  form.querySelector("#selection-total").textContent =
    `${state.draft.devices.length} / ${total}`;
  form.querySelector('button[type="submit"]').disabled =
    state.draft.devices.length !== total || !total;
  form.querySelectorAll(".device-choice").forEach((choice) => {
    const index = state.draft.devices.indexOf(
      choice.querySelector("input").value,
    );
    choice.querySelector(".badge").textContent =
      index >= 0 ? `Rank ${index}` : "—";
  });
});

document.addEventListener("change", (event) => {
  if (
    event.target.name !== "workflow" ||
    !event.target.closest('[data-form="training"]')
  )
    return;
  const smoke = event.target.value === "smoke";
  Object.assign(state.draft.config, {
    workflow: event.target.value,
    model: smoke ? "tiny-random-bert" : "prajjwal1/bert-mini",
    dataset: smoke ? "synthetic" : "rajpurkar/squad",
  });
  render();
});

setInterval(async () => {
  if (!state.auth || state.refreshing || dialog.open) return;
  state.refreshing = true;
  try {
    if (state.screen === "clusters") await refreshClusters();
    else {
      await refreshCluster();
      if (state.screen === "job") await fetchJob(state.job.id);
    }
    const editing =
      document.activeElement?.matches("input,select,textarea") ||
      document.activeElement?.closest("dialog");
    if (!editing) {
      const log = document.querySelector("#run-log");
      const follow =
        log && log.scrollTop + log.clientHeight >= log.scrollHeight - 25;
      const scroll = log?.scrollTop || 0;
      render();
      const current = document.querySelector("#run-log");
      if (current) current.scrollTop = follow ? current.scrollHeight : scroll;
    }
  } catch (error) {
    if (state.auth) toast(error.message, true);
  } finally {
    state.refreshing = false;
  }
}, 2500);

async function initialize() {
  try {
    state.auth = await api("/api/auth/me");
    state.catalog = await api("/api/catalog");
    await refreshClusters();
  } catch {
    state.auth = null;
  }
  render();
}
initialize();
