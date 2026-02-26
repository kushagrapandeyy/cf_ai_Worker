import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { useAgent } from "agents/react";
import { useAgentChat } from "@cloudflare/ai-chat/react";
import type { UIMessage } from "ai";

type ToolPart = {
    type: "tool";
    toolCallId: string;
    toolName: string;
    state: "pending" | "approval-required" | "rejected" | "output-available";
    input?: Record<string, unknown>;
    output?: unknown;
};

const SendIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="22" y1="2" x2="11" y2="13" />
        <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
);

const TrashIcon = () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14H6L5 6" /><path d="M10 11v6" /><path d="M14 11v6" /><path d="M9 6V4h6v2" />
    </svg>
);

const BotIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="11" width="18" height="10" rx="2" /><circle cx="12" cy="5" r="2" /><path d="M12 7v4" /><line x1="8" y1="16" x2="8" y2="16" /><line x1="16" y1="16" x2="16" y2="16" />
    </svg>
);

const UserIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
    </svg>
);

const ChevronIcon = ({ open }: { open: boolean }) => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
        style={{ transform: open ? "rotate(90deg)" : "rotate(0deg)", transition: "transform 0.2s" }}>
        <polyline points="9 18 15 12 9 6" />
    </svg>
);

function ToolCard({ part }: { part: ToolPart; onApprove?: () => void; onReject?: () => void }) {
    const [open, setOpen] = useState(false);

    const toolLabels: Record<string, string> = {
        searchWeb: "🔍 Web Search",
        getUserInfo: "🌐 Browser Info",
        setReminder: "⏰ Set Reminder",
    };

    const label = toolLabels[part.toolName] ?? part.toolName;

    if (part.state === "pending") {
        return (
            <div className="tool-card tool-pending">
                <span className="tool-label">{label}</span>
                <span className="tool-status-dot" />
                <span className="tool-status-text">Running…</span>
            </div>
        );
    }

    if (part.state === "output-available") {
        return (
            <div className="tool-card tool-done" onClick={() => setOpen(!open)} role="button" tabIndex={0}
                onKeyDown={(e) => e.key === "Enter" && setOpen(!open)}>
                <span className="tool-label">{label}</span>
                <span className="tool-badge success">Done</span>
                <ChevronIcon open={open} />
                {open && (
                    <pre className="tool-output">{JSON.stringify(part.output, null, 2)}</pre>
                )}
            </div>
        );
    }

    return null;
}

function ApprovalCard({
    part,
    onApprove,
    onReject,
}: {
    part: ToolPart;
    onApprove: () => void;
    onReject: () => void;
}) {
    const toolLabels: Record<string, string> = {
        setReminder: "⏰ Set Reminder",
    };
    const label = toolLabels[part.toolName] ?? part.toolName;

    return (
        <div className="approval-card">
            <div className="approval-header">
                <span className="approval-title">Approval Required</span>
                <span className="tool-badge warning">Pending</span>
            </div>
            <p className="approval-desc">
                Sage wants to run <strong>{label}</strong> with these parameters:
            </p>
            <pre className="tool-output">{JSON.stringify(part.input, null, 2)}</pre>
            <div className="approval-actions">
                <button className="btn-approve" onClick={onApprove}>✓ Approve</button>
                <button className="btn-reject" onClick={onReject}>✕ Reject</button>
            </div>
        </div>
    );
}

function MessageBubble({
    msg,
    addToolApprovalResponse,
}: {
    msg: UIMessage;
    addToolApprovalResponse: (r: { id: string; approved: boolean }) => void;
}) {
    const isUser = msg.role === "user";

    return (
        <div className={`message-row ${isUser ? "message-row--user" : "message-row--assistant"}`}>
            <div className={`avatar ${isUser ? "avatar--user" : "avatar--bot"}`}>
                {isUser ? <UserIcon /> : <BotIcon />}
            </div>
            <div className="message-bubble-group">
                {msg.parts.map((part, i) => {
                    if (part.type === "text") {
                        return (
                            <div key={i} className={`bubble ${isUser ? "bubble--user" : "bubble--assistant"}`}>
                                <span className="bubble-text">{part.text}</span>
                            </div>
                        );
                    }

                    const tp = part as unknown as ToolPart;

                    if (tp.type === "tool" && tp.state === "approval-required") {
                        return (
                            <ApprovalCard
                                key={tp.toolCallId}
                                part={tp}
                                onApprove={() =>
                                    addToolApprovalResponse({ id: tp.toolCallId, approved: true })
                                }
                                onReject={() =>
                                    addToolApprovalResponse({ id: tp.toolCallId, approved: false })
                                }
                            />
                        );
                    }

                    if (tp.type === "tool") {
                        return <ToolCard key={tp.toolCallId} part={tp} />;
                    }

                    return null;
                })}
            </div>
        </div>
    );
}

function TypingIndicator() {
    return (
        <div className="message-row message-row--assistant">
            <div className="avatar avatar--bot"><BotIcon /></div>
            <div className="bubble bubble--assistant typing-indicator">
                <span /><span /><span />
            </div>
        </div>
    );
}

function Chat() {
    const agent = useAgent({ agent: "ChatAgent" });
    const bottomRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const [inputValue, setInputValue] = useState("");

    const {
        messages,
        sendMessage,
        clearHistory,
        addToolApprovalResponse,
        status,
    } = useAgentChat({
        agent,
        onToolCall: async ({ toolCall, addToolOutput }) => {
            if (toolCall.toolName === "getUserInfo") {
                addToolOutput({
                    toolCallId: toolCall.toolCallId,
                    output: {
                        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
                        locale: navigator.language,
                        localTime: new Date().toLocaleTimeString(),
                        userAgent: navigator.userAgent.split(" ").slice(-2).join(" "),
                    },
                });
            }
        },
    });

    const isStreaming = status === "streaming";

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    function handleSend(e: React.FormEvent) {
        e.preventDefault();
        const text = inputValue.trim();
        if (!text || isStreaming) return;

        sendMessage({ text });
        setInputValue("");
        inputRef.current?.focus();
    }

    const suggestedPrompts = [
        "Search for the latest on AI agents",
        "What timezone am I in?",
        "Remind me to take a break in 60 seconds",
        "Explain how Cloudflare Durable Objects work",
    ];

    return (
        <div className="app">
            <aside className="sidebar">
                <div className="sidebar-logo">
                    <span className="logo-glyph">✦</span>
                    <span className="logo-name">Sage</span>
                </div>
                <p className="sidebar-tagline">AI Research Assistant</p>
                <div className="sidebar-divider" />
                <p className="sidebar-section-label">Try asking</p>
                <div className="suggested-prompts">
                    {suggestedPrompts.map((p) => (
                        <button
                            key={p}
                            className="suggested-prompt"
                            disabled={isStreaming}
                            onClick={() => sendMessage({ text: p })}
                        >
                            {p}
                        </button>
                    ))}
                </div>
                <div className="sidebar-spacer" />
                <div className="sidebar-footer">
                    <div className="model-badge">
                        <span className="model-dot" />
                        Llama 3.1 8B
                    </div>
                    <button className="clear-btn" onClick={clearHistory} title="Clear history">
                        <TrashIcon />
                        Clear history
                    </button>
                </div>
            </aside>

            <main className="main">
                <div className="messages-area">
                    {messages.length === 0 ? (
                        <div className="empty-state">
                            <div className="empty-icon">✦</div>
                            <h2>Hello, I'm Sage</h2>
                            <p>Your AI research assistant on Cloudflare. Ask me anything, I can search the web, check your timezone, and even set reminders.</p>
                        </div>
                    ) : (
                        messages.map((msg) => (
                            <MessageBubble
                                key={msg.id}
                                msg={msg}
                                addToolApprovalResponse={addToolApprovalResponse}
                            />
                        ))
                    )}
                    {isStreaming && <TypingIndicator />}
                    <div ref={bottomRef} />
                </div>

                <form className="input-bar" onSubmit={handleSend}>
                    <input
                        ref={inputRef}
                        className="input-field"
                        type="text"
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        placeholder="Ask Sage anything…"
                        disabled={isStreaming}
                        autoComplete="off"
                    />
                    <button
                        className={`send-btn ${isStreaming ? "send-btn--disabled" : ""}`}
                        type="submit"
                        disabled={isStreaming || !inputValue.trim()}
                        aria-label="Send message"
                    >
                        <SendIcon />
                    </button>
                </form>
            </main>
        </div>
    );
}

const root = createRoot(document.getElementById("root")!);
root.render(<Chat />);
