import { create } from 'zustand'
import {
  getHealth, agentsAPI, modelsAPI, toolsAPI, assessmentsAPI,
  tasksAPI, findingsAPI, reportsAPI, workflowsAPI, conversationsAPI,
  auditAPI, approvalsAPI,
} from '../api'

export const useStore = create((set, get) => ({
  // Language
  lang: 'en',
  setLang: (lang) => set({ lang }),

  // Reduced motion
  reducedMotion: false,
  setReducedMotion: (v) => set({ reducedMotion: v }),

  // Sidebar
  sidebarCollapsed: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),

  // Data caches
  agents: [],
  models: [],
  tools: [],
  assessments: [],
  tasks: [],
  findings: [],
  reports: [],
  workflows: [],
  conversations: [],
  auditEvents: [],
  approvals: [],
  health: null,

  // Actions
  fetchAll: async () => {
    await Promise.all([
      get().fetchHealth(),
      get().fetchAgents(),
      get().fetchModels(),
      get().fetchTools(),
      get().fetchAssessments(),
      get().fetchFindings(),
      get().fetchWorkflows(),
      get().fetchConversations(),
      get().fetchAudit(),
    ])
  },

  fetchHealth: async () => {
    try {
      const { data } = await getHealth()
      set({ health: data })
    } catch { set({ health: { status: 'error', backend: 'offline' } }) }
  },

  fetchAgents: async () => {
    try { const { data } = await agentsAPI.list(); set({ agents: data }) } catch {}
  },
  fetchModels: async () => {
    try { const { data } = await modelsAPI.list(); set({ models: data }) } catch {}
  },
  fetchTools: async () => {
    try { const { data } = await toolsAPI.list(); set({ tools: data }) } catch {}
  },
  fetchAssessments: async () => {
    try { const { data } = await assessmentsAPI.list(); set({ assessments: data }) } catch {}
  },
  fetchTasks: async (assessmentId) => {
    try { const { data } = await tasksAPI.list(assessmentId); set({ tasks: data }) } catch {}
  },
  fetchFindings: async () => {
    try { const { data } = await findingsAPI.list(); set({ findings: data }) } catch {}
  },
  fetchReports: async (assessmentId) => {
    try { const { data } = await reportsAPI.list(assessmentId); set({ reports: data }) } catch {}
  },
  fetchWorkflows: async () => {
    try { const { data } = await workflowsAPI.list(); set({ workflows: data }) } catch {}
  },
  fetchConversations: async () => {
    try { const { data } = await conversationsAPI.list(); set({ conversations: data }) } catch {}
  },
  fetchAudit: async () => {
    try { const { data } = await auditAPI.list(50); set({ auditEvents: data }) } catch {}
  },
  fetchApprovals: async (status) => {
    try { const { data } = await approvalsAPI.list(status); set({ approvals: data }) } catch {}
  },
}))
