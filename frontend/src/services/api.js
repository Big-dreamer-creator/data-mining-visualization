const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8001/api";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`);
  }
  return response.json();
}

export function fetchAlgorithms() {
  return request("/algorithms");
}

export function fetchDatasets() {
  return request("/datasets");
}

export function fetchDataset(id) {
  return request(`/datasets/${id}`);
}

export function fetchAlgorithm(id) {
  return request(`/algorithms/${id}`);
}

export function runVisualization(id, payload) {
  return request(`/algorithms/${id}/visualization`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function evaluateAlgorithm(id, payload) {
  return request(`/algorithms/${id}/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export const runAlgorithm = runVisualization;
