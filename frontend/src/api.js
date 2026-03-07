// frontend/src/api.js
// All communication with the FastAPI backend lives here.

const BASE_URL = '/api/v1'

/**
 * Sends article text to the backend for prediction.
 * @param {string} text
 * @returns {Promise<{ id, prediction, confidence, created_at }>}
 */
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

/**
 * Fetches prediction history from the backend.
 * @param {number} limit
 * @param {number} offset
 * @returns {Promise<{ total, items }>}
 */
export async function fetchHistory(limit = 20, offset = 0) {
  const response = await fetch(`${BASE_URL}/history?limit=${limit}&offset=${offset}`)

  if (!response.ok) {
    throw new Error(`Failed to fetch history (${response.status})`)
  }

  return response.json()
}
