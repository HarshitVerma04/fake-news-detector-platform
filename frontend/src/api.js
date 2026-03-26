// frontend/src/api.js

const BASE_URL = '/api/v1'

export async function predictArticle(text) {
  const response = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error(err.detail || `Request failed (${response.status})`)
  }
  return response.json()
}

export async function fetchHistory(limit = 20, offset = 0) {
  const response = await fetch(`${BASE_URL}/history?limit=${limit}&offset=${offset}`)
  if (!response.ok) throw new Error(`Failed to fetch history (${response.status})`)
  return response.json()
}

export async function fetchModelInfo() {
  const response = await fetch(`${BASE_URL}/models`)
  if (!response.ok) throw new Error(`Failed to fetch model info (${response.status})`)
  return response.json()
}
