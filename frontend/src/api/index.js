import axios from 'axios'

// Use public URL for browser requests (preview proxy), fallback to local
const API_URL = import.meta.env.VITE_PUBLIC_API_URL || import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 30000,
})

// --- Health ---
export const getHealth = () => api.get('/api/health')

// --- Agents ---
export const agentsAPI = {
  list: () => api.get('/api/agents'),
  get: (id) => api.get(`/api/agents/${id}`),
  create: (data) => api.post('/api/agents', data),
  update: (id, data) => api.put(`/api/agents/${id}`, data),
  delete: (id) => api.delete(`/api/agents/${id}`),
  test: (id) => api.post(`/api/agents/${id}/test`),
}

// --- Models ---
export const modelsAPI = {
  list: () => api.get('/api/models'),
  get: (id) => api.get(`/api/models/${id}`),
  create: (data) => api.post('/api/models', data),
  update: (id, data) => api.put(`/api/models/${id}`, data),
  delete: (id) => api.delete(`/api/models/${id}`),
  test: (id) => api.post(`/api/models/${id}/test`),
  activate: (id) => api.post(`/api/models/${id}/activate`),
}

// --- Tools ---
export const toolsAPI = {
  list: () => api.get('/api/tools'),
  get: (id) => api.get(`/api/tools/${id}`),
  create: (data) => api.post('/api/tools', data),
  update: (id, data) => api.put(`/api/tools/${id}`, data),
  delete: (id) => api.delete(`/api/tools/${id}`),
  check: (id) => api.post(`/api/tools/${id}/check`),
}

// --- Assessments ---
export const assessmentsAPI = {
  list: () => api.get('/api/assessments'),
  get: (id) => api.get(`/api/assessments/${id}`),
  create: (data) => api.post('/api/assessments', data),
  update: (id, data) => api.put(`/api/assessments/${id}`, data),
  delete: (id) => api.delete(`/api/assessments/${id}`),
  validateScope: (id, target) => api.post(`/api/assessments/${id}/validate-scope`, null, { params: { target } }),
}

// --- Tasks ---
export const tasksAPI = {
  list: (assessmentId) => api.get('/api/tasks', { params: { assessment_id: assessmentId } }),
  get: (id) => api.get(`/api/tasks/${id}`),
  create: (data) => api.post('/api/tasks', data),
  update: (id, data) => api.put(`/api/tasks/${id}`, data),
  cancel: (id) => api.post(`/api/tasks/${id}/cancel`),
}

// --- Conversations ---
export const conversationsAPI = {
  list: () => api.get('/api/conversations'),
  get: (id) => api.get(`/api/conversations/${id}`),
  create: (data) => api.get('/api/conversations'), // placeholder
  newConversation: (data) => api.post('/api/conversations', data),
  update: (id, data) => api.put(`/api/conversations/${id}`, data),
  delete: (id) => api.delete(`/api/conversations/${id}`),
  sendMessage: (id, content) => api.post(`/api/conversations/${id}/messages`, null, { params: { content } }),
}

// --- Workflows ---
export const workflowsAPI = {
  list: () => api.get('/api/workflows'),
  get: (id) => api.get(`/api/workflows/${id}`),
  create: (data) => api.post('/api/workflows', data),
  update: (id, data) => api.put(`/api/workflows/${id}`, data),
  delete: (id) => api.delete(`/api/workflows/${id}`),
  duplicate: (id) => api.post(`/api/workflows/${id}/duplicate`),
}

// --- Findings ---
export const findingsAPI = {
  list: (params) => api.get('/api/findings', { params }),
  get: (id) => api.get(`/api/findings/${id}`),
  create: (data) => api.post('/api/findings', data),
  update: (id, data) => api.put(`/api/findings/${id}`, data),
  delete: (id) => api.delete(`/api/findings/${id}`),
}

// --- Reports ---
export const reportsAPI = {
  list: (assessmentId) => api.get('/api/reports', { params: { assessment_id: assessmentId } }),
  get: (id) => api.get(`/api/reports/${id}`),
  create: (data) => api.post('/api/reports', data),
  generate: (assessmentId) => api.post(`/api/reports/generate/${assessmentId}`),
  update: (id, data) => api.put(`/api/reports/${id}`, data),
  delete: (id) => api.delete(`/api/reports/${id}`),
}

// --- Audit ---
export const auditAPI = {
  list: (limit) => api.get('/api/audit', { params: { limit } }),
}

// --- Approvals ---
export const approvalsAPI = {
  list: (status) => api.get('/api/approvals', { params: { status } }),
  create: (data) => api.post('/api/approvals', data),
  approve: (id) => api.post(`/api/approvals/${id}/approve`),
  reject: (id) => api.post(`/api/approvals/${id}/reject`),
}

export default api
