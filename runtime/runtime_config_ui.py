def render_runtime_config_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenFabric Runtime Config</title>
  <style>
    :root {
      --bg: #0b0d10;
      --panel: #12181f;
      --panel-strong: #18212b;
      --line: rgba(255, 255, 255, 0.09);
      --line-strong: rgba(255, 255, 255, 0.18);
      --text: #f3f5f7;
      --muted: #9aa6b2;
      --accent: #38bdf8;
      --accent-soft: rgba(56, 189, 248, 0.14);
      --danger: #fb7185;
      --success: #34d399;
      --shadow: 0 26px 80px rgba(0, 0, 0, 0.34);
      --radius-xl: 26px;
      --radius-lg: 18px;
      --radius-md: 14px;
    }

    * {
      box-sizing: border-box;
    }

    html, body {
      margin: 0;
      min-height: 100%;
      background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.09), transparent 24%),
        radial-gradient(circle at top right, rgba(29, 78, 216, 0.08), transparent 26%),
        linear-gradient(180deg, #0b0d10 0%, #0d1117 100%);
      color: var(--text);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
    }

    body {
      padding: 28px;
    }

    .shell {
      max-width: 1080px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }

    .hero {
      display: grid;
      gap: 14px;
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: var(--radius-xl);
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.015)),
        var(--panel);
      box-shadow: var(--shadow);
    }

    .eyebrow {
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }

    h1 {
      margin: 0;
      font-size: 42px;
      line-height: 1;
      font-family: "IBM Plex Sans Condensed", "Avenir Next Condensed", sans-serif;
    }

    .hero p {
      margin: 0;
      max-width: 72ch;
      color: var(--muted);
      line-height: 1.7;
    }

    .hero-actions,
    .toolbar {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    .button,
    .link-button {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 11px 16px;
      cursor: pointer;
      font: inherit;
      color: var(--text);
      background: rgba(255, 255, 255, 0.03);
      transition: transform 140ms ease, border-color 140ms ease, background 140ms ease;
      text-decoration: none;
    }

    .button:hover,
    .link-button:hover {
      transform: translateY(-1px);
      border-color: var(--line-strong);
    }

    .button.primary {
      background: linear-gradient(180deg, rgba(56, 189, 248, 0.18), rgba(56, 189, 248, 0.08));
      border-color: rgba(56, 189, 248, 0.28);
    }

    .toolbar {
      justify-content: space-between;
      padding: 0 4px;
      color: var(--muted);
      font-size: 13px;
    }

    .status {
      min-height: 22px;
      color: var(--muted);
    }

    .status.success { color: var(--success); }
    .status.error { color: var(--danger); }

    .card-grid {
      display: grid;
      gap: 14px;
    }

    .config-card {
      display: grid;
      gap: 14px;
      padding: 22px;
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.015)),
        var(--panel-strong);
      box-shadow: 0 18px 44px rgba(0, 0, 0, 0.16);
    }

    .config-top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
    }

    .config-top h2 {
      margin: 0 0 8px;
      font-size: 24px;
      font-family: "IBM Plex Sans Condensed", "Avenir Next Condensed", sans-serif;
    }

    .config-top p {
      margin: 0;
      color: var(--muted);
      line-height: 1.65;
      max-width: 68ch;
    }

    .config-meta {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
    }

    .pill {
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.03);
    }

    .toggle {
      position: relative;
      width: 68px;
      height: 38px;
      border-radius: 999px;
      border: 1px solid var(--line-strong);
      background: rgba(255, 255, 255, 0.05);
      cursor: pointer;
      padding: 0;
      flex: 0 0 auto;
    }

    .toggle[aria-checked="true"] {
      background: linear-gradient(180deg, rgba(56, 189, 248, 0.24), rgba(56, 189, 248, 0.14));
      border-color: rgba(56, 189, 248, 0.36);
    }

    .toggle-knob {
      position: absolute;
      top: 4px;
      left: 4px;
      width: 28px;
      height: 28px;
      border-radius: 999px;
      background: #f3f5f7;
      transition: transform 150ms ease;
    }

    .toggle[aria-checked="true"] .toggle-knob {
      transform: translateX(28px);
    }

    .config-path {
      padding: 16px 18px;
      border: 1px dashed rgba(255, 255, 255, 0.14);
      border-radius: var(--radius-lg);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.65;
      background: rgba(255, 255, 255, 0.02);
    }

    code {
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.95em;
      color: #eef2f7;
    }

    @media (max-width: 720px) {
      body {
        padding: 18px;
      }

      .hero,
      .config-card {
        padding: 20px;
      }

      .config-top {
        flex-direction: column;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">OpenFabric Runtime Control</div>
      <h1>Live Runtime Configuration</h1>
      <p>
        Toggle noisy traces and logging behavior without restarting the local stack. These settings are
        stored in a shared runtime config file so the gateway and local agents can pick them up live.
      </p>
      <div class="hero-actions">
        <a class="link-button" href="/">Open chat UI</a>
        <button id="refresh-config" class="button" type="button">Refresh</button>
        <button id="reset-config" class="button primary" type="button">Reset to defaults</button>
      </div>
    </section>

    <div class="toolbar">
      <div id="status" class="status">Loading current runtime settings...</div>
      <div id="updated-at"></div>
    </div>

    <section id="config-cards" class="card-grid"></section>
    <section id="config-path" class="config-path"></section>
  </div>

  <script>
    const state = {
      settings: [],
      configPath: '',
      updatedAt: '',
    };

    function escapeHtml(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function setStatus(message, tone = '') {
      const node = document.getElementById('status');
      node.textContent = message;
      node.className = tone ? `status ${tone}` : 'status';
    }

    function formatDate(value) {
      if (!value) return 'No runtime overrides saved yet';
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      return `Last updated ${date.toLocaleString()}`;
    }

    function render() {
      const cards = document.getElementById('config-cards');
      cards.innerHTML = state.settings.map((setting) => `
        <article class="config-card">
          <div class="config-top">
            <div>
              <h2>${escapeHtml(setting.label)}</h2>
              <p>${escapeHtml(setting.description)}</p>
            </div>
            <button
              class="toggle"
              type="button"
              role="switch"
              aria-checked="${setting.value ? 'true' : 'false'}"
              data-setting-id="${escapeHtml(setting.id)}"
            >
              <span class="toggle-knob"></span>
            </button>
          </div>
          <div class="config-meta">
            <div class="pill">Scope: ${escapeHtml(setting.scope)}</div>
            <div class="pill">Env default: <code>${escapeHtml(setting.env_var)}=${escapeHtml(String(setting.effective_default))}</code></div>
            <div class="pill">Factory default: <code>${escapeHtml(String(setting.default))}</code></div>
          </div>
        </article>
      `).join('');

      document.getElementById('updated-at').textContent = formatDate(state.updatedAt);
      document.getElementById('config-path').innerHTML = `
        Shared config file: <code>${escapeHtml(state.configPath || 'Unavailable')}</code><br />
        Changes apply immediately to processes that read the shared runtime config on each log or stream decision.
      `;

      for (const button of document.querySelectorAll('[data-setting-id]')) {
        button.addEventListener('click', async () => {
          const id = button.getAttribute('data-setting-id');
          const current = state.settings.find((item) => item.id === id);
          if (!current) return;
          await saveSetting(id, !current.value);
        });
      }
    }

    async function loadConfig() {
      const response = await fetch('/api/runtime-config');
      if (!response.ok) {
        throw new Error(`Failed to load runtime config: ${response.status}`);
      }
      const payload = await response.json();
      state.settings = Array.isArray(payload.settings) ? payload.settings : [];
      state.configPath = payload.config_path || '';
      state.updatedAt = payload.updated_at || '';
      render();
    }

    async function saveSetting(id, value) {
      setStatus(`Saving ${id}...`);
      const response = await fetch('/api/runtime-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ values: { [id]: value } }),
      });
      if (!response.ok) {
        const text = await response.text();
        setStatus(text || `Failed to save ${id}`, 'error');
        return;
      }
      const payload = await response.json();
      state.settings = Array.isArray(payload.settings) ? payload.settings : [];
      state.configPath = payload.config_path || '';
      state.updatedAt = payload.updated_at || '';
      render();
      setStatus(`Saved ${id}.`, 'success');
    }

    async function resetSettings() {
      setStatus('Resetting runtime config...');
      const response = await fetch('/api/runtime-config/reset', {
        method: 'POST',
      });
      if (!response.ok) {
        const text = await response.text();
        setStatus(text || 'Failed to reset runtime config.', 'error');
        return;
      }
      const payload = await response.json();
      state.settings = Array.isArray(payload.settings) ? payload.settings : [];
      state.configPath = payload.config_path || '';
      state.updatedAt = payload.updated_at || '';
      render();
      setStatus('Runtime config reset to defaults.', 'success');
    }

    document.getElementById('refresh-config').addEventListener('click', async () => {
      try {
        setStatus('Refreshing runtime config...');
        await loadConfig();
        setStatus('Runtime config refreshed.', 'success');
      } catch (error) {
        console.error(error);
        setStatus(String(error.message || error), 'error');
      }
    });

    document.getElementById('reset-config').addEventListener('click', async () => {
      await resetSettings();
    });

    loadConfig()
      .then(() => setStatus('Runtime config ready.', 'success'))
      .catch((error) => {
        console.error(error);
        setStatus(String(error.message || error), 'error');
      });
  </script>
</body>
</html>"""
