import argparse
import os

import openwebui_gateway
from web_compat import HTMLResponse


DEFAULT_UI_PORT = 8310


def render_codex_ui_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Codex UI for OpenFabric</title>
  <style>
    :root {
      --bg: #0b0d10;
      --bg-elevated: #11161c;
      --bg-panel: #151b22;
      --bg-panel-strong: #1b222b;
      --bg-input: #0f141a;
      --line: rgba(255, 255, 255, 0.09);
      --line-strong: rgba(255, 255, 255, 0.16);
      --text: #f3f5f7;
      --muted: #9aa6b2;
      --accent: #7dd3fc;
      --accent-strong: #38bdf8;
      --accent-soft: rgba(56, 189, 248, 0.14);
      --user: #e7eef7;
      --assistant: #f6f7f8;
      --danger: #fb7185;
      --shadow: 0 28px 80px rgba(0, 0, 0, 0.34);
      --sidebar-width: 300px;
      --content-width: 920px;
      --radius-xl: 24px;
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
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.08), transparent 26%),
        radial-gradient(circle at top right, rgba(99, 102, 241, 0.08), transparent 28%),
        linear-gradient(180deg, #0b0d10 0%, #0d1117 100%);
      color: var(--text);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
    }

    body {
      display: grid;
      grid-template-columns: var(--sidebar-width) minmax(0, 1fr);
    }

    button, input, textarea, select {
      font: inherit;
    }

    .sidebar {
      min-height: 100vh;
      border-right: 1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.02), transparent 22%),
        rgba(9, 12, 16, 0.84);
      backdrop-filter: blur(18px);
      display: flex;
      flex-direction: column;
    }

    .brand {
      padding: 20px 18px 16px;
      border-bottom: 1px solid var(--line);
    }

    .brand-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .brand-title {
      font-family: "IBM Plex Sans Condensed", "Avenir Next Condensed", sans-serif;
      font-size: 28px;
      letter-spacing: 0.02em;
      font-weight: 700;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 11px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.03);
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }

    .brand-subtitle {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }

    .sidebar-actions {
      padding: 16px 18px 12px;
      display: grid;
      gap: 10px;
      border-bottom: 1px solid var(--line);
    }

    .primary-button,
    .secondary-button,
    .icon-button,
    .ghost-button {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.03);
      color: var(--text);
      cursor: pointer;
      transition: transform 140ms ease, border-color 140ms ease, background 140ms ease;
    }

    .primary-button:hover,
    .secondary-button:hover,
    .icon-button:hover,
    .ghost-button:hover {
      transform: translateY(-1px);
      border-color: var(--line-strong);
    }

    .primary-button {
      width: 100%;
      padding: 12px 14px;
      background: linear-gradient(180deg, rgba(56, 189, 248, 0.16), rgba(56, 189, 248, 0.08));
      border-color: rgba(56, 189, 248, 0.26);
      font-weight: 600;
    }

    .sidebar-meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .meta-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
      background: rgba(255, 255, 255, 0.02);
      min-height: 78px;
    }

    .meta-label {
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .meta-value {
      margin-top: 7px;
      font-size: 20px;
      font-weight: 700;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
    }

    .meta-detail {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }

    .session-shell {
      flex: 1;
      overflow: auto;
      padding: 12px;
      display: grid;
      gap: 10px;
    }

    .session-card {
      border: 1px solid transparent;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.02);
      padding: 14px;
      text-align: left;
      color: var(--text);
      cursor: pointer;
      transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;
    }

    .session-card:hover {
      background: rgba(255, 255, 255, 0.05);
      border-color: rgba(255, 255, 255, 0.08);
      transform: translateY(-1px);
    }

    .session-card.active {
      background: linear-gradient(180deg, rgba(56, 189, 248, 0.12), rgba(56, 189, 248, 0.05));
      border-color: rgba(56, 189, 248, 0.26);
    }

    .session-title {
      font-size: 15px;
      font-weight: 600;
      line-height: 1.35;
    }

    .session-snippet {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    .session-meta {
      margin-top: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 11px;
    }

    .main {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
    }

    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(10, 13, 17, 0.78);
      backdrop-filter: blur(18px);
    }

    .topbar h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 650;
      letter-spacing: 0.01em;
    }

    .topbar p {
      margin: 5px 0 0;
      color: var(--muted);
      font-size: 13px;
    }

    .topbar-controls {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    .select-shell,
    .toggle-shell {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.03);
      color: var(--muted);
      min-height: 46px;
    }

    .select-shell label,
    .toggle-shell label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }

    select {
      border: 0;
      outline: none;
      background: transparent;
      color: var(--text);
      min-width: 180px;
    }

    .chat-scroll {
      overflow: auto;
      padding: 28px 24px 20px;
    }

    .chat-frame {
      max-width: var(--content-width);
      margin: 0 auto;
      display: grid;
      gap: 18px;
      min-height: 100%;
    }

    .welcome-shell {
      display: grid;
      gap: 22px;
      padding: 28px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.02)),
        var(--bg-elevated);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: var(--shadow);
    }

    .welcome-shell h2 {
      margin: 0;
      font-size: 42px;
      line-height: 1;
      font-weight: 700;
      font-family: "IBM Plex Sans Condensed", "Avenir Next Condensed", sans-serif;
    }

    .welcome-shell p {
      margin: 0;
      color: var(--muted);
      max-width: 66ch;
      line-height: 1.65;
    }

    .prompt-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .prompt-card {
      padding: 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.03);
      color: var(--text);
      text-align: left;
      cursor: pointer;
    }

    .prompt-card strong {
      display: block;
      font-size: 15px;
      margin-bottom: 8px;
    }

    .prompt-card span {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }

    .message-row {
      display: grid;
      gap: 8px;
    }

    .message-meta {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
      padding: 0 4px;
    }

    .message-card {
      border: 1px solid var(--line);
      border-radius: var(--radius-xl);
      background: var(--bg-panel);
      padding: 20px 22px;
      overflow: hidden;
      line-height: 1.7;
      box-shadow: 0 18px 44px rgba(0, 0, 0, 0.18);
    }

    .message-row.user {
      justify-items: end;
    }

    .message-row.user .message-card {
      max-width: min(84%, 720px);
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.03)),
        #171d25;
      color: var(--user);
    }

    .message-row.assistant .message-card {
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.015)),
        var(--bg-panel-strong);
      color: var(--assistant);
    }

    .message-content > :first-child {
      margin-top: 0;
    }

    .message-content > :last-child {
      margin-bottom: 0;
    }

    .message-content p {
      margin: 0 0 1em;
    }

    .message-content h1,
    .message-content h2,
    .message-content h3 {
      margin: 1.1em 0 0.55em;
      line-height: 1.18;
      font-family: "IBM Plex Sans Condensed", "Avenir Next Condensed", sans-serif;
    }

    .message-content h1 { font-size: 28px; }
    .message-content h2 { font-size: 24px; }
    .message-content h3 { font-size: 20px; }

    .message-content ul,
    .message-content ol {
      margin: 0 0 1em 1.35em;
      padding: 0;
    }

    .message-content li {
      margin: 0.35em 0;
    }

    .message-content a {
      color: var(--accent);
      text-decoration: none;
    }

    .message-content a:hover {
      text-decoration: underline;
    }

    .message-content code {
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.94em;
      padding: 0.14em 0.38em;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.08);
    }

    .codeblock {
      margin: 1em 0;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 18px;
      background: #0f141a;
      overflow: hidden;
    }

    .codeblock-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }

    .codeblock pre {
      margin: 0;
      padding: 16px 18px 18px;
      overflow: auto;
      color: #eef2f7;
      font-size: 13px;
      line-height: 1.65;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
    }

    .streaming-indicator {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }

    .composer-shell {
      border-top: 1px solid var(--line);
      background: rgba(10, 13, 17, 0.84);
      backdrop-filter: blur(18px);
      padding: 16px 24px 22px;
    }

    .composer {
      max-width: calc(var(--content-width) + 32px);
      margin: 0 auto;
      border: 1px solid var(--line);
      border-radius: 26px;
      background: var(--bg-input);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .composer textarea {
      width: 100%;
      min-height: 104px;
      resize: vertical;
      border: 0;
      outline: none;
      background: transparent;
      color: var(--text);
      padding: 18px 20px 16px;
      line-height: 1.6;
      font-family: "IBM Plex Sans", "Avenir Next", sans-serif;
    }

    .composer-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px 16px;
      border-top: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.02);
    }

    .composer-hint {
      color: var(--muted);
      font-size: 12px;
    }

    .composer-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .secondary-button,
    .ghost-button,
    .icon-button {
      padding: 10px 14px;
      color: var(--muted);
    }

    .icon-button {
      min-width: 44px;
      min-height: 44px;
      padding: 0;
      display: inline-grid;
      place-items: center;
      border-radius: 14px;
    }

    .secondary-button {
      min-height: 44px;
    }

    .ghost-button {
      background: transparent;
    }

    .empty-sidebar {
      border: 1px dashed rgba(255, 255, 255, 0.12);
      border-radius: 18px;
      padding: 16px;
      text-align: center;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }

    .error-banner {
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(251, 113, 133, 0.25);
      background: rgba(251, 113, 133, 0.08);
      color: #ffd5dc;
      font-size: 13px;
      line-height: 1.55;
    }

    @media (max-width: 1080px) {
      body {
        grid-template-columns: 1fr;
      }

      .sidebar {
        min-height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }

      .session-shell {
        max-height: 280px;
      }
    }

    @media (max-width: 720px) {
      .topbar,
      .chat-scroll,
      .composer-shell {
        padding-left: 16px;
        padding-right: 16px;
      }

      .prompt-grid {
        grid-template-columns: 1fr;
      }

      .message-row.user .message-card {
        max-width: 100%;
      }

      .topbar {
        align-items: flex-start;
        flex-direction: column;
      }

      .topbar-controls {
        width: 100%;
      }

      .select-shell {
        width: 100%;
      }

      select {
        min-width: 0;
        flex: 1;
      }
    }
  </style>
</head>
<body>
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-top">
        <div class="brand-title">Codex</div>
        <div class="status-pill">
          <span id="api-status">API ready</span>
        </div>
      </div>
      <div class="brand-subtitle">
        Standalone OpenFabric chat UI using the same OpenAI-compatible API surface as Open WebUI.
      </div>
    </div>

    <div class="sidebar-actions">
      <button id="new-chat" class="primary-button" type="button">New chat</button>
      <div class="sidebar-meta">
        <div class="meta-card">
          <div class="meta-label">Model</div>
          <div id="sidebar-model" class="meta-value">...</div>
          <div class="meta-detail">Loaded from <code>/v1/models</code></div>
        </div>
        <div class="meta-card">
          <div class="meta-label">Sessions</div>
          <div id="session-count" class="meta-value">0</div>
          <div class="meta-detail">Stored locally in this browser</div>
        </div>
      </div>
    </div>

    <div id="session-list" class="session-shell"></div>
  </aside>

  <main class="main">
    <header class="topbar">
      <div>
        <h1 id="chat-title">New chat</h1>
        <p id="chat-subtitle">OpenFabric Codex-style chat shell over the existing gateway contract.</p>
      </div>
      <div class="topbar-controls">
        <div class="select-shell">
          <label for="model-select">Model</label>
          <select id="model-select"></select>
        </div>
        <button id="open-config" class="ghost-button" type="button">Runtime config</button>
        <button id="clear-chat" class="ghost-button" type="button">Clear chat</button>
      </div>
    </header>

    <section class="chat-scroll" id="chat-scroll">
      <div class="chat-frame" id="chat-frame"></div>
    </section>

    <section class="composer-shell">
      <div class="composer">
        <textarea
          id="composer-input"
          placeholder="Ask anything. This UI sends the same chat completion payload Open WebUI would use."
        ></textarea>
        <div class="composer-footer">
          <div class="composer-hint">
            <span>Enter to send</span>
            <span> · </span>
            <span>Shift+Enter for a newline</span>
          </div>
          <div class="composer-actions">
            <button id="stop-stream" class="secondary-button" type="button">Stop</button>
            <button id="send-message" class="primary-button" type="button">Send</button>
          </div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const STORAGE_KEY = 'openfabric_codex_ui_sessions_v1';
    const DEFAULT_SYSTEM_PROMPTS = [
      {
        title: 'Explore the runtime graph',
        description: 'Ask the agent to explain the current workflow graph, validators, routers, and reducers.'
      },
      {
        title: 'Investigate a run',
        description: 'Ask for help understanding a persisted run, its retries, and its validation outcomes.'
      },
      {
        title: 'Plan a new feature',
        description: 'Use the same gateway contract, but drive a custom implementation task through the agent stack.'
      },
      {
        title: 'Debug a failing request',
        description: 'Have the agent trace a request path and identify where the workflow condensed or retried.'
      }
    ];

    const state = {
      sessions: [],
      activeSessionId: null,
      models: [],
      model: '',
      loadingModels: false,
      streaming: false,
      abortController: null,
    };

    function uid(prefix) {
      return `${prefix}_${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36)}`;
    }

    function escapeHtml(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function formatTime(timestamp) {
      try {
        return new Date(timestamp).toLocaleString();
      } catch (_error) {
        return '';
      }
    }

    function truncate(value, length = 120) {
      const text = String(value || '').trim().replace(/\\s+/g, ' ');
      if (text.length <= length) return text;
      return `${text.slice(0, length - 3)}...`;
    }

    function makeSession() {
      const now = new Date().toISOString();
      return {
        id: uid('session'),
        title: 'New chat',
        createdAt: now,
        updatedAt: now,
        model: state.model || '',
        messages: [],
      };
    }

    function activeSession() {
      return state.sessions.find((session) => session.id === state.activeSessionId) || null;
    }

    function setStatus(text, healthy = true) {
      const el = document.getElementById('api-status');
      el.textContent = text;
      el.style.color = healthy ? 'var(--muted)' : 'var(--danger)';
    }

    function loadSessions() {
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        const parsed = raw ? JSON.parse(raw) : [];
        state.sessions = Array.isArray(parsed) ? parsed : [];
      } catch (_error) {
        state.sessions = [];
      }
      if (!state.sessions.length) {
        const session = makeSession();
        state.sessions = [session];
      }
      if (!state.activeSessionId || !state.sessions.some((session) => session.id === state.activeSessionId)) {
        state.activeSessionId = state.sessions[0].id;
      }
    }

    function persistSessions() {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state.sessions));
      document.getElementById('session-count').textContent = String(state.sessions.length);
    }

    function updateSessionTitle(session) {
      if (!session) return;
      const firstUser = session.messages.find((message) => message.role === 'user' && message.content.trim());
      session.title = firstUser ? truncate(firstUser.content, 48) : 'New chat';
      session.updatedAt = new Date().toISOString();
      if (!session.model && state.model) {
        session.model = state.model;
      }
    }

    function createNewChat(seedPrompt = '') {
      const session = makeSession();
      if (state.model) {
        session.model = state.model;
      }
      state.sessions.unshift(session);
      state.activeSessionId = session.id;
      persistSessions();
      renderSidebar();
      renderChat();
      const composer = document.getElementById('composer-input');
      composer.value = seedPrompt;
      composer.focus();
    }

    function clearActiveChat() {
      const session = activeSession();
      if (!session) return;
      session.messages = [];
      updateSessionTitle(session);
      persistSessions();
      renderSidebar();
      renderChat();
    }

    function switchSession(sessionId) {
      if (!state.sessions.some((session) => session.id === sessionId)) return;
      state.activeSessionId = sessionId;
      renderSidebar();
      renderChat();
    }

    function removeSession(sessionId) {
      const next = state.sessions.filter((session) => session.id !== sessionId);
      state.sessions = next.length ? next : [makeSession()];
      if (!state.sessions.some((session) => session.id === state.activeSessionId)) {
        state.activeSessionId = state.sessions[0].id;
      }
      persistSessions();
      renderSidebar();
      renderChat();
    }

    function renderSidebar() {
      const shell = document.getElementById('session-list');
      const model = state.model || 'n/a';
      document.getElementById('sidebar-model').textContent = truncate(model, 10);
      document.getElementById('session-count').textContent = String(state.sessions.length);

      if (!state.sessions.length) {
        shell.innerHTML = '<div class="empty-sidebar">No saved chats yet.</div>';
        return;
      }

      shell.innerHTML = state.sessions.map((session) => {
        const active = session.id === state.activeSessionId ? 'active' : '';
        const snippetSource = session.messages.length
          ? session.messages[session.messages.length - 1].content
          : 'Fresh conversation';
        const modelChip = session.model ? escapeHtml(session.model) : 'gateway';
        return `
          <div class="session-card ${active}" data-session-id="${escapeHtml(session.id)}">
            <div class="session-title">${escapeHtml(session.title || 'New chat')}</div>
            <div class="session-snippet">${escapeHtml(truncate(snippetSource, 88) || 'Fresh conversation')}</div>
            <div class="session-meta">
              <span>${escapeHtml(modelChip)}</span>
              <span>${escapeHtml(formatTime(session.updatedAt))}</span>
            </div>
          </div>
        `;
      }).join('');

      Array.from(shell.querySelectorAll('.session-card')).forEach((card) => {
        card.addEventListener('click', () => switchSession(card.dataset.sessionId));
      });
    }

    function renderWelcome() {
      return `
        <section class="welcome-shell">
          <div>
            <h2>Codex-style chat for OpenFabric</h2>
            <p>
              This standalone UI speaks the same OpenAI-compatible chat completion API as Open WebUI.
              You can keep using the same backend contract while replacing the presentation layer.
            </p>
          </div>
          <div class="prompt-grid">
            ${DEFAULT_SYSTEM_PROMPTS.map((prompt) => `
              <button class="prompt-card" type="button" data-seed-prompt="${escapeHtml(prompt.title)}: ${escapeHtml(prompt.description)}">
                <strong>${escapeHtml(prompt.title)}</strong>
                <span>${escapeHtml(prompt.description)}</span>
              </button>
            `).join('')}
          </div>
        </section>
      `;
    }

    function renderInlineFormatting(text) {
      return text
        .replace(/`([^`\\n]+)`/g, '<code>$1</code>')
        .replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>')
        .replace(/\\[([^\\]]+)\\]\\((https?:\\/\\/[^\\s)]+)\\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
    }

    function renderTextBlock(text) {
      const escaped = escapeHtml(text);
      const chunks = escaped.split(/\\n{2,}/).filter(Boolean);
      return chunks.map((chunk) => {
        const lines = chunk.split('\\n');
        if (lines.every((line) => /^\\s*[-*]\\s+/.test(line))) {
          return `<ul>${lines.map((line) => `<li>${renderInlineFormatting(line.replace(/^\\s*[-*]\\s+/, ''))}</li>`).join('')}</ul>`;
        }
        return lines.map((line) => {
          const compact = line.trim();
          if (!compact) return '';
          if (compact.startsWith('### ')) return `<h3>${renderInlineFormatting(compact.slice(4))}</h3>`;
          if (compact.startsWith('## ')) return `<h2>${renderInlineFormatting(compact.slice(3))}</h2>`;
          if (compact.startsWith('# ')) return `<h1>${renderInlineFormatting(compact.slice(2))}</h1>`;
          return `<p>${renderInlineFormatting(compact)}</p>`;
        }).join('');
      }).join('');
    }

    function renderMarkdown(content) {
      const text = String(content || '');
      const segments = [];
      const codePattern = /```([a-zA-Z0-9_+-]*)?\\n([\\s\\S]*?)```/g;
      let cursor = 0;
      let match;
      while ((match = codePattern.exec(text)) !== null) {
        if (match.index > cursor) {
          segments.push({ type: 'text', value: text.slice(cursor, match.index) });
        }
        segments.push({ type: 'code', lang: match[1] || 'text', value: match[2] || '' });
        cursor = codePattern.lastIndex;
      }
      if (cursor < text.length) {
        segments.push({ type: 'text', value: text.slice(cursor) });
      }
      return segments.map((segment) => {
        if (segment.type === 'code') {
          return `
            <div class="codeblock">
              <div class="codeblock-header">
                <span>${escapeHtml(segment.lang || 'text')}</span>
                <span>code</span>
              </div>
              <pre><code>${escapeHtml(segment.value)}</code></pre>
            </div>
          `;
        }
        return renderTextBlock(segment.value);
      }).join('');
    }

    function renderChat() {
      const frame = document.getElementById('chat-frame');
      const session = activeSession();
      document.getElementById('chat-title').textContent = session?.title || 'New chat';
      document.getElementById('chat-subtitle').textContent = session?.messages.length
        ? 'Streaming chat over /v1/chat/completions'
        : 'Same backend contract, custom presentation layer.';

      if (!session || !session.messages.length) {
        frame.innerHTML = renderWelcome();
        Array.from(frame.querySelectorAll('.prompt-card')).forEach((button) => {
          button.addEventListener('click', () => {
            document.getElementById('composer-input').value = button.dataset.seedPrompt || '';
            document.getElementById('composer-input').focus();
          });
        });
        return;
      }

      frame.innerHTML = session.messages.map((message) => {
        const roleClass = message.role === 'user' ? 'user' : 'assistant';
        const roleLabel = message.role === 'user' ? 'You' : 'Assistant';
        const metaRight = message.streaming
          ? '<span class="streaming-indicator">streaming</span>'
          : `<span>${escapeHtml(formatTime(message.createdAt || session.updatedAt))}</span>`;
        return `
          <article class="message-row ${roleClass}">
            <div class="message-meta">
              <span>${escapeHtml(roleLabel)}</span>
              ${metaRight}
            </div>
            <div class="message-card">
              <div class="message-content">${renderMarkdown(message.content || '')}</div>
            </div>
          </article>
        `;
      }).join('');

      window.requestAnimationFrame(() => {
        const scroll = document.getElementById('chat-scroll');
        scroll.scrollTop = scroll.scrollHeight;
      });
    }

    function setComposerEnabled(enabled) {
      document.getElementById('composer-input').disabled = !enabled;
      document.getElementById('send-message').disabled = !enabled;
      document.getElementById('stop-stream').disabled = enabled;
    }

    async function loadModels() {
      if (state.loadingModels) return;
      state.loadingModels = true;
      try {
        const response = await fetch('/v1/models');
        if (!response.ok) {
          throw new Error(`Model request failed: ${response.status}`);
        }
        const payload = await response.json();
        state.models = Array.isArray(payload.data) ? payload.data : [];
        state.model = state.model || state.models[0]?.id || 'openfabric-planner';
        const select = document.getElementById('model-select');
        select.innerHTML = state.models.map((model) => `
          <option value="${escapeHtml(model.id)}">${escapeHtml(model.id)}</option>
        `).join('') || '<option value="openfabric-planner">openfabric-planner</option>';
        select.value = state.model;
        setStatus('API ready', true);
      } catch (error) {
        setStatus('API unavailable', false);
        document.getElementById('model-select').innerHTML = '<option value="openfabric-planner">openfabric-planner</option>';
        state.model = 'openfabric-planner';
        console.error(error);
      } finally {
        state.loadingModels = false;
        renderSidebar();
      }
    }

    function buildApiMessages(session) {
      return session.messages
        .filter((message) => (message.role === 'user' || message.role === 'assistant') && String(message.content || '').trim())
        .map((message) => ({ role: message.role, content: message.content }));
    }

    function decodeSseChunk(rawChunk) {
      return rawChunk
        .split('\\n')
        .map((line) => line.trim())
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trim())
        .filter(Boolean);
    }

    async function streamCompletion(session, assistantMessage) {
      const body = {
        model: state.model || 'openfabric-planner',
        messages: buildApiMessages(session),
        stream: true,
      };

      state.abortController = new AbortController();
      state.streaming = true;
      setComposerEnabled(false);

      try {
        const response = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: state.abortController.signal,
        });

        if (!response.ok || !response.body) {
          const text = await response.text();
          throw new Error(text || `Completion request failed: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\\n\\n');
          buffer = parts.pop() || '';
          for (const part of parts) {
            for (const item of decodeSseChunk(part)) {
              if (item === '[DONE]') {
                assistantMessage.streaming = false;
                updateSessionTitle(session);
                persistSessions();
                renderSidebar();
                renderChat();
                return;
              }
              try {
                const payload = JSON.parse(item);
                const delta = payload?.choices?.[0]?.delta?.content;
                if (typeof delta === 'string') {
                  assistantMessage.content += delta;
                  assistantMessage.streaming = true;
                  session.updatedAt = new Date().toISOString();
                  persistSessions();
                  renderChat();
                }
              } catch (error) {
                console.error('Failed to parse SSE chunk', error, item);
              }
            }
          }
        }
      } finally {
        assistantMessage.streaming = false;
        state.streaming = false;
        state.abortController = null;
        setComposerEnabled(true);
        updateSessionTitle(session);
        persistSessions();
        renderSidebar();
        renderChat();
      }
    }

    async function sendMessage() {
      if (state.streaming) return;
      const composer = document.getElementById('composer-input');
      const content = composer.value.trim();
      if (!content) return;

      const session = activeSession();
      if (!session) return;

      const now = new Date().toISOString();
      const userMessage = { role: 'user', content, createdAt: now };
      const assistantMessage = { role: 'assistant', content: '', createdAt: now, streaming: true };
      session.messages.push(userMessage, assistantMessage);
      session.model = state.model || session.model || 'openfabric-planner';
      updateSessionTitle(session);
      persistSessions();
      renderSidebar();
      renderChat();
      composer.value = '';

      try {
        await streamCompletion(session, assistantMessage);
      } catch (error) {
        assistantMessage.streaming = false;
        assistantMessage.content = `### Request failed\\n\\n${String(error.message || error)}`;
        updateSessionTitle(session);
        persistSessions();
        renderSidebar();
        renderChat();
        setStatus('Request failed', false);
      }
    }

    function stopStreaming() {
      if (!state.abortController) return;
      state.abortController.abort();
      state.abortController = null;
      state.streaming = false;
      setComposerEnabled(true);
      setStatus('Generation stopped', false);
    }

    function wireEvents() {
      document.getElementById('new-chat').addEventListener('click', () => createNewChat());
      document.getElementById('clear-chat').addEventListener('click', clearActiveChat);
      document.getElementById('open-config').addEventListener('click', () => {
        window.open('/config', '_blank', 'noopener');
      });
      document.getElementById('send-message').addEventListener('click', sendMessage);
      document.getElementById('stop-stream').addEventListener('click', stopStreaming);
      document.getElementById('model-select').addEventListener('change', (event) => {
        state.model = event.target.value;
        const session = activeSession();
        if (session) {
          session.model = state.model;
          persistSessions();
          renderSidebar();
        }
      });
      document.getElementById('composer-input').addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
          event.preventDefault();
          sendMessage();
        }
      });
    }

    async function boot() {
      loadSessions();
      persistSessions();
      renderSidebar();
      renderChat();
      wireEvents();
      setComposerEnabled(true);
      await loadModels();
    }

    boot().catch((error) => {
      console.error(error);
      setStatus('UI boot failed', false);
      document.getElementById('chat-frame').innerHTML = `
        <div class="error-banner">
          Failed to start the Codex-style UI shell. ${escapeHtml(String(error.message || error))}
        </div>
      `;
    });
  </script>
</body>
</html>"""


def create_app(spec_path: str = openwebui_gateway.DEFAULT_SPEC_PATH, timeout_seconds: float | None = 300, selected_agents: list[str] | None = None, enable_context: bool = False):
    if enable_context:
        os.environ["ENABLE_CONTEXT"] = "1"

    app = openwebui_gateway.create_app(spec_path, timeout_seconds=timeout_seconds, selected_agents=selected_agents)
    if getattr(app, "_codex_ui_registered", False):
        return app

    @app.get("/", response_class=HTMLResponse)
    def root():
        return HTMLResponse(render_codex_ui_html())

    @app.get("/ui", response_class=HTMLResponse)
    def ui():
        return HTMLResponse(render_codex_ui_html())

    setattr(app, "_codex_ui_registered", True)
    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=openwebui_gateway.DEFAULT_SPEC_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_UI_PORT)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--agent",
        action="append",
        dest="selected_agents",
        help="Start only the named agent(s). Repeat the flag or pass a comma-separated list.",
    )
    parser.add_argument(
        "--enable-context",
        action="store_true",
        help="Include prior chat history in planner requests.",
    )
    args = parser.parse_args()

    uvicorn.run(
        create_app(
            spec_path=args.spec,
            timeout_seconds=args.timeout,
            selected_agents=args.selected_agents,
            enable_context=args.enable_context,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
