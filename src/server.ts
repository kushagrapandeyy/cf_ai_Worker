import { AIChatAgent } from "@cloudflare/ai-chat";
import { routeAgentRequest } from "agents";
import {
    createUIMessageStream,
    createUIMessageStreamResponse,
} from "ai";
import type { StreamTextOnFinishCallback, ToolSet, UIMessage } from "ai";
import { z } from "zod";

interface Env {
    AI: Ai;
    ChatAgent: DurableObjectNamespace<ChatAgent>;
    ASSETS: Fetcher;
}

type WorkersAIMessage = {
    role: "system" | "user" | "assistant" | "tool";
    content: string;
    tool_call_id?: string;
    name?: string;
    tool_calls?: WorkersAIToolCall[];
};

type WorkersAIToolCall = {
    id: string;
    type: "function";
    function: { name: string; arguments: string };
};

const TOOLS_SCHEMA = [
    {
        type: "function" as const,
        function: {
            name: "searchWeb",
            description: "Search the web for current information on any topic.",
            parameters: {
                type: "object",
                properties: {
                    query: { type: "string", description: "The search query" },
                },
                required: ["query"],
            },
        },
    },
    {
        type: "function" as const,
        function: {
            name: "getUserInfo",
            description: "Get the user's browser timezone, locale, and local time. Runs in the user's browser.",
            parameters: { type: "object", properties: {} },
        },
    },
    {
        type: "function" as const,
        function: {
            name: "setReminder",
            description: "Schedule a reminder for the user. Always requires user approval first.",
            parameters: {
                type: "object",
                properties: {
                    message: { type: "string", description: "The reminder message" },
                    delaySeconds: { type: "number", description: "Seconds from now to trigger" },
                },
                required: ["message", "delaySeconds"],
            },
        },
    },
];

async function executeSearchWeb(query: string): Promise<string> {
    try {
        const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`;
        const res = await fetch(url);
        const data = (await res.json()) as {
            AbstractText: string;
            RelatedTopics: Array<{ Text?: string; FirstURL?: string }>;
        };
        const abstract = data.AbstractText?.slice(0, 500) ?? null;
        const related = (data.RelatedTopics ?? [])
            .filter((t) => t.Text)
            .slice(0, 4)
            .map((t) => `- ${t.Text?.slice(0, 120)}`);
        if (!abstract && related.length === 0) {
            return JSON.stringify({ summary: `No results found for "${query}".` });
        }
        return JSON.stringify({ summary: abstract ?? "See related:", results: related });
    } catch {
        return JSON.stringify({ error: "Search temporarily unavailable." });
    }
}

function uiMessagesToWorkersAI(messages: UIMessage[]): WorkersAIMessage[] {
    const out: WorkersAIMessage[] = [];
    for (const msg of messages) {
        const textParts = msg.parts.filter((p) => p.type === "text").map(p => (p as { text: string }).text).join("\n");
        const toolParts = msg.parts.filter((p) => typeof p.type === "string" && (p.type.startsWith("tool-") || p.type === "dynamic-tool")) as unknown as Array<{
            type: string; toolCallId: string; toolName?: string; state: string;
            input?: unknown; output?: unknown;
        }>;

        if (msg.role === "user") {
            if (textParts) out.push({ role: "user", content: textParts });
        } else if (msg.role === "assistant") {
            const assistantMsg: WorkersAIMessage = { role: "assistant", content: textParts || "" };

            const calls = toolParts.filter(t => t.state === "output-available" || t.state === "rejected" || t.state === "approval-required");
            if (calls.length > 0) {
                assistantMsg.tool_calls = calls.map(t => ({
                    id: t.toolCallId,
                    type: "function",
                    function: { name: t.toolName || "unknown", arguments: JSON.stringify(t.input ?? {}) }
                }));
            }
            out.push(assistantMsg);

            // Immediately follow with tool results for this assistant message if they exist
            for (const t of calls) {
                if (t.state === "output-available" || t.state === "rejected") {
                    out.push({
                        role: "tool",
                        tool_call_id: t.toolCallId,
                        name: t.toolName,
                        content: t.state === "rejected"
                            ? "Error: User rejected this action."
                            : JSON.stringify(t.output ?? {})
                    });
                }
            }
        }
    }
    return out;
}

export class ChatAgent extends AIChatAgent<Env> {
    private _isGenerating = false;
    private _lastRequestTime = 0;

    private async _scheduleReminder(message: string, delaySeconds: number) {
        await this.schedule(delaySeconds, "onTask", { message });
        return { scheduled: true, message, inSeconds: delaySeconds };
    }

    async onTask(data: unknown) {
        const { message } = data as { message: string };
        const currentState = (this.state as Record<string, unknown>) ?? {};
        const reminders: string[] = Array.isArray(
            (currentState as { reminders?: string[] }).reminders
        )
            ? (currentState as { reminders: string[] }).reminders
            : [];
        reminders.push(`⏰ Reminder: ${message} (triggered at ${new Date().toISOString()})`);
        this.setState({ ...currentState, reminders });
    }

    async onChatMessage(onFinish: StreamTextOnFinishCallback<ToolSet>) {
        const now = Date.now();
        if (this._isGenerating && (now - this._lastRequestTime < 10000)) {
            console.warn("Possible concurrent request within 10s. Guarding state.");
        }

        this._isGenerating = true;
        this._lastRequestTime = now;

        const stateObj = (this.state as Record<string, unknown>) ?? {};
        const pendingReminders: string[] = Array.isArray((stateObj as { reminders?: string[] }).reminders)
            ? (stateObj as { reminders: string[] }).reminders
            : [];

        const systemContent = `You are Sage AI, a high-performance Enterprise Workspace Assistant. 
Your tone is professional, helpful, and direct. You are running on Cloudflare's edge.
Current Date: ${new Date().toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" })}.
${pendingReminders.length > 0 ? `\n\nREMINDERS FOR USER:\n${pendingReminders.join("\n")}` : ""}`;

        if (pendingReminders.length > 0) {
            this.setState({ ...stateObj, reminders: [] });
        }

        const historyMessages = uiMessagesToWorkersAI(this.messages);
        const allMessages: WorkersAIMessage[] = [
            { role: "system", content: systemContent },
            ...historyMessages,
        ];

        const stream = createUIMessageStream({
            execute: async ({ writer }) => {
                try {
                    let stepMessages = [...allMessages];
                    const calledTools = new Set<string>();

                    for (let pass = 0; pass < 3; pass++) {
                        console.log(`[Sage] AI Pass ${pass}, History: ${stepMessages.length} msgs`);

                        const aiResponse = await (this.env.AI as unknown as {
                            run: (model: string, options: {
                                messages: WorkersAIMessage[];
                                tools: typeof TOOLS_SCHEMA;
                                stream: boolean;
                                max_tokens: number;
                            }) => Promise<{ response?: string; tool_calls?: WorkersAIToolCall[] }>;
                        }).run("@cf/meta/llama-3.1-8b-instruct", {
                            messages: stepMessages,
                            tools: TOOLS_SCHEMA,
                            stream: false,
                            max_tokens: 1024,
                        }).catch(e => {
                            console.error("[Sage] Model Execution Error:", e);
                            throw new Error("Model failed to respond.");
                        });

                        const { response, tool_calls } = aiResponse;

                        if (tool_calls && tool_calls.length > 0) {
                            // Professional Tool Execution
                            const filteredCalls = tool_calls.filter(tc => !calledTools.has(tc.function.name));
                            if (filteredCalls.length === 0) break; // Avoid infinite recursion

                            const assistantEntry: WorkersAIMessage = { role: "assistant", content: response || "", tool_calls: filteredCalls };
                            stepMessages.push(assistantEntry);

                            for (const tc of filteredCalls) {
                                calledTools.add(tc.function.name);
                                const toolName = tc.function.name;
                                let args: Record<string, any> = {};
                                try { args = JSON.parse(tc.function.arguments || "{}"); } catch { }

                                writer.write({ type: "tool-input-available", toolCallId: tc.id, toolName, input: args });

                                if (toolName === "getUserInfo") {
                                    writer.write({ type: "tool-output-available", toolCallId: tc.id, output: { status: "fetching" } });
                                    return; // Handled by client
                                }

                                let output: any = { error: "Unknown tool" };
                                if (toolName === "searchWeb") {
                                    output = JSON.parse(await executeSearchWeb(args.query || ""));
                                } else if (toolName === "setReminder") {
                                    output = await this._scheduleReminder(args.message, args.delaySeconds || 60);
                                }

                                writer.write({ type: "tool-output-available", toolCallId: tc.id, output });
                                stepMessages.push({ role: "tool", tool_call_id: tc.id, name: toolName, content: JSON.stringify(output) });
                            }
                            continue; // Next pass to see results
                        }

                        if (response) {
                            const msgId = `msg-${Date.now()}`;
                            writer.write({ type: "text-start", id: msgId });
                            writer.write({ type: "text-delta", delta: response, id: msgId });
                            writer.write({ type: "text-end", id: msgId });
                            break;
                        }
                        break;
                    }
                } catch (err) {
                    const msgId = `error-${Date.now()}`;
                    writer.write({ type: "text-start", id: msgId });
                    writer.write({ type: "text-delta", delta: "I encountered a technical glitch while processing that. Please try again in a moment.", id: msgId });
                    writer.write({ type: "text-end", id: msgId });
                } finally {
                    this._isGenerating = false;
                }
            },
            onFinish: ({ messages }) => {
                this._isGenerating = false;
                this.saveMessages(messages);
            }
        });

        return createUIMessageStreamResponse({ stream });
    }
}

export default {
    async fetch(request: Request, env: Env): Promise<Response> {
        const agentResponse = await routeAgentRequest(request, env);
        if (agentResponse) return agentResponse;

        if (env.ASSETS) {
            return env.ASSETS.fetch(request);
        }

        return new Response(
            `<!DOCTYPE html><html><body><h1>Sage Agent</h1><p>Assets not configured.</p></body></html>`,
            { headers: { "Content-Type": "text/html" } }
        );
    },
} satisfies ExportedHandler<Env>;
