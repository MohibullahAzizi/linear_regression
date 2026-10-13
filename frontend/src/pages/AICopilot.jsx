import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, Send, MessageSquare, Bot, User, Trash2, ChevronDown } from 'lucide-react'
import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'
import { PageHeader, Card, EmptyState } from '../components/UI'
import { conversationsAPI } from '../api'

export default function AICopilot() {
  const { t } = useI18n()
  const conversations = useStore((s) => s.conversations)
  const fetchConversations = useStore((s) => s.fetchConversations)
  const agents = useStore((s) => s.agents)
  const models = useStore((s) => s.models)
  const [activeConv, setActiveConv] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [convData, setConvData] = useState(null)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    if (activeConv) loadConversation(activeConv)
  }, [activeConv])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const loadConversation = async (id) => {
    try {
      const { data } = await conversationsAPI.get(id)
      setMessages(data.messages || [])
      setConvData(data)
    } catch { /* */ }
  }

  const newConversation = async () => {
    try {
      const { data } = await conversationsAPI.newConversation({ title: t('newConversation') })
      fetchConversations()
      setActiveConv(data.id)
      setMessages([])
      setConvData(data)
    } catch { /* */ }
  }

  const send = async () => {
    if (!input.trim() || !activeConv || sending) return
    const content = input.trim()
    setInput('')
    setSending(true)
    // Optimistic add
    setMessages((m) => [...m, { role: 'user', content, created_at: new Date().toISOString() }])
    try {
      const { data } = await conversationsAPI.sendMessage(activeConv, content)
      setMessages((m) => [...m, data.assistant_message])
      fetchConversations()
    } catch (e) {
      setMessages((m) => [...m, { role: 'assistant', content: 'Error: Could not send message. ' + e.message, metadata: { source: 'error' } }])
    }
    setSending(false)
  }

  const delConv = async (id) => {
    if (!confirm('Delete this conversation?')) return
    await conversationsAPI.delete(id)
    if (activeConv === id) { setActiveConv(null); setMessages([]) }
    fetchConversations()
  }

  return (
    <div>
      <PageHeader title={t('aiCopilot')} subtitle="Natural-language control with the Master Security Agent" />

      <div className="flex gap-4 h-[calc(100vh-200px)]">
        {/* Conversation list */}
        <div className="w-64 shrink-0 space-y-2 overflow-y-auto">
          <button onClick={newConversation} className="btn-primary w-full flex items-center justify-center gap-2 text-sm mb-2">
            <Plus className="w-4 h-4" /> {t('newConversation')}
          </button>
          {conversations.length === 0 ? (
            <p className="text-xs text-text-secondary text-center py-4">{t('noData')}</p>
          ) : conversations.map((c) => (
            <div
              key={c.id}
              onClick={() => setActiveConv(c.id)}
              className={`p-3 rounded-lg cursor-pointer transition-colors group ${activeConv === c.id ? 'bg-accent-primary/10 border border-accent-primary/30' : 'card hover:bg-bg-elevated'}`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <MessageSquare className="w-3.5 h-3.5 text-text-secondary shrink-0" />
                  <span className="text-xs text-text-primary truncate">{c.title}</span>
                </div>
                <button onClick={(e) => { e.stopPropagation(); delConv(c.id) }} className="opacity-0 group-hover:opacity-100 text-text-secondary hover:text-error transition-all">
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Chat area */}
        <div className="flex-1 flex flex-col card rounded-xl overflow-hidden">
          {!activeConv ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <Bot className="w-16 h-16 text-accent-primary/30 mx-auto mb-4" />
                <p className="text-text-secondary text-sm">{t('newConversation')} to start chatting with the AI Copilot</p>
              </div>
            </div>
          ) : (
            <>
              {/* Chat header */}
              <div className="px-4 py-3 border-b border-border flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-primary to-accent-secondary flex items-center justify-center">
                    <Bot className="w-4 h-4 text-bg-main" />
                  </div>
                  <div>
                    <div className="text-sm font-medium text-text-primary">Master Security Agent</div>
                    <div className="text-[10px] text-text-secondary">
                      {convData?.selected_agent || 'No agent selected'} • Model: {models.find((m) => m.is_active)?.display_name || 'None'}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {convData?.selected_model && <span className="text-[10px] text-accent-secondary">{convData.selected_model}</span>}
                </div>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 ? (
                  <div className="text-center py-8">
                    <p className="text-sm text-text-secondary">Send a message to start the conversation.</p>
                    <p className="text-xs text-text-secondary mt-2">Try: "Explain the current assessment" or "Select CAI agent"</p>
                  </div>
                ) : messages.map((msg, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                  >
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${msg.role === 'user' ? 'bg-accent-secondary/20' : 'bg-gradient-to-br from-accent-primary to-accent-tertiary'}`}>
                      {msg.role === 'user' ? <User className="w-4 h-4 text-accent-secondary" /> : <Bot className="w-4 h-4 text-bg-main" />}
                    </div>
                    <div className={`max-w-[70%] ${msg.role === 'user' ? 'text-right' : ''}`}>
                      <div className={`rounded-xl px-4 py-3 ${msg.role === 'user' ? 'bg-accent-secondary/15 text-text-primary' : 'bg-bg-elevated text-text-primary border border-border'}`}>
                        <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                      </div>
                      {msg.metadata?.source && msg.metadata.source !== 'local' && (
                        <div className="text-[10px] text-text-secondary mt-1 px-2">
                          via {msg.metadata.source === 'ai_model' ? `AI: ${msg.metadata.model}` : msg.metadata.source}
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
                {sending && (
                  <div className="flex gap-3">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-primary to-accent-tertiary flex items-center justify-center">
                      <Bot className="w-4 h-4 text-bg-main" />
                    </div>
                    <div className="rounded-xl px-4 py-3 bg-bg-elevated border border-border">
                      <div className="flex gap-1">
                        <span className="w-2 h-2 rounded-full bg-accent-primary animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="w-2 h-2 rounded-full bg-accent-primary animate-bounce" style={{ animationDelay: '150ms' }} />
                        <span className="w-2 h-2 rounded-full bg-accent-primary animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <div className="p-4 border-t border-border">
                <div className="flex gap-2 items-end">
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
                    placeholder={t('typeMessage')}
                    rows="1"
                    className="input-field resize-none flex-1"
                    style={{ minHeight: '44px', maxHeight: '120px' }}
                  />
                  <button onClick={send} disabled={!input.trim() || sending} className="btn-primary shrink-0 h-11 px-4">
                    <Send className="w-4 h-4" />
                  </button>
                </div>
                <p className="text-[10px] text-text-secondary mt-2">Press Enter to send • Shift+Enter for new line</p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
