// frontend/src/api.js

const BASE_URL = '/api/v1'

function authHeaders(token) {
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

async function handleResponse(res) {
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Request failed (${res.status})`)
  }
  return res.json()
}

// ── Auth ──────────────────────────────────────
export async function registerUser({ username, email, password }) {
  return handleResponse(await fetch(`${BASE_URL}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password }),
  }))
}

export async function loginUser({ username, password }) {
  const form = new URLSearchParams({ username, password })
  return handleResponse(await fetch(`${BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: form,
  }))
}

export async function refreshToken(refresh_token) {
  return handleResponse(await fetch(`${BASE_URL}/auth/refresh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token }),
  }))
}

export async function getMe(token) {
  return handleResponse(await fetch(`${BASE_URL}/auth/me`, {
    headers: authHeaders(token),
  }))
}

export async function requestPasswordReset(email) {
  return handleResponse(await fetch(`${BASE_URL}/auth/password-reset/request`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email }),
  }))
}

export async function confirmPasswordReset({ token, new_password }) {
  return handleResponse(await fetch(`${BASE_URL}/auth/password-reset/confirm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token, new_password }),
  }))
}

// ── Predict ───────────────────────────────────
export async function predictArticle(text, token) {
  return handleResponse(await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: authHeaders(token),
    body: JSON.stringify({ text }),
  }))
}

export async function fetchFromUrl(url, token) {
  return handleResponse(await fetch(`${BASE_URL}/fetch-url`, {
    method: 'POST',
    headers: authHeaders(token),
    body: JSON.stringify({ url }),
  }))
}

// ── History ───────────────────────────────────
export async function fetchHistory(limit = 20, offset = 0, token) {
  return handleResponse(await fetch(`${BASE_URL}/history?limit=${limit}&offset=${offset}`, {
    headers: authHeaders(token),
  }))
}

// ── Stats ─────────────────────────────────────
export async function fetchStats() {
  return handleResponse(await fetch(`${BASE_URL}/stats`))
}

// ── Models ────────────────────────────────────
export async function fetchModelInfo() {
  return handleResponse(await fetch(`${BASE_URL}/models`))
}
