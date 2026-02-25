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
    ChatAgent: DurableObjectNamespace;
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
        const textParts = msg.parts.filter((p) => p.type === "text");
        const toolParts = msg.parts.filter((p) => typeof p.type === "string" && (p.type.startsWith("tool-") || p.type === "dynamic-tool")) as unknown as Array<{
            type: string; toolCallId: string; toolName?: string; state: string;
            input?: unknown; output?: unknown;
        }>;

        if (msg.role === "user") {
            const text = textParts.map((p) => (p as { text: string }).text).join("\n");
            if (text) out.push({ role: "user", content: text });
            for (const t of toolParts) {
                if (t.state === "output-available" || t.state === "rejected") {
                    out.push({
                        role: "tool",
                        tool_call_id: t.toolCallId,
                        name: t.toolName,
                        content: t.state === "rejected"
                            ? "Tool execution was rejected by the user."
                            : JSON.stringify(t.output),
                    });
                }
            }
        } else if (msg.role === "assistant") {
            const text = textParts.map((p) => (p as { text: string }).text).join("\n");
            if (text) out.push({ role: "assistant", content: text });
            const calls = toolParts.filter((t) =>
                t.state === "output-available" || t.state === "approval-required" || t.state === "rejected"
            );
            if (calls.length > 0) {
                out.push({
                    role: "assistant",
                    content: "",
                    tool_calls: calls.map((t) => ({
                        id: t.toolCallId,
                        type: "function",
                        function: { name: t.toolName || "unknown", arguments: JSON.stringify(t.input ?? {}) },
                    })),
                });
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
        if (this._isGenerating || now - this._lastRequestTime < 3000) {
            console.warn("Blocked concurrent or rapid request. Rate limiting to protect APIs.");
            const stream = createUIMessageStream({
                execute: async ({ writer }) => {
                    const msgId = `msg-${Date.now()}`;
                    writer.write({ type: "text-start", id: msgId });
                    writer.write({ type: "text-delta", delta: "Rate limit: Please wait a few seconds before sending another message.", id: msgId });
                    writer.write({ type: "text-end", id: msgId });
                }
            });
            return createUIMessageStreamResponse({ stream });
        }

        this._isGenerating = true;
        this._lastRequestTime = now;

        const stateObj = (this.state as Record<string, unknown>) ?? {};
        const pendingReminders: string[] = Array.isArray(
            (stateObj as { reminders?: string[] }).reminders
        )
            ? (stateObj as { reminders: string[] }).reminders
            : [];

        let systemContent = `You are Sage, a brilliant AI research assistant on Cloudflare.
Use your tools to answer questions. Use searchWeb only when the user explicitly asks for current, live, or real-time information.
Do not call searchWeb if the question can be answered from general knowledge.
Never call the same tool more than once per user message.
Call getUserInfo when asked about timezone, location, locale, or browser info.
Call setReminder when asked to set a reminder.
Today: ${new Date().toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" })}.`;

        if (pendingReminders.length > 0) {
            systemContent += `\n\nACTIVE REMINDERS:\n${pendingReminders.join("\n")}\nInform the user.`;
            this.setState({ ...stateObj, reminders: [] });
        }

        const historyMessages = uiMessagesToWorkersAI(this.messages);
        const allMessages: WorkersAIMessage[] = [
            { role: "system", content: systemContent },
            ...historyMessages,
        ];

        const stream = createUIMessageStream({
            execute: async ({ writer }) => {
                let pendingToolCalls: WorkersAIToolCall[] = [];
                let stepMessages = [...allMessages];
                let assistantText = "";
                const toolResults: Array<{ toolCallId: string; toolName: string; result: unknown }> = [];
                const calledToolNames = new Set<string>();

                for (let step = 0; step < 2; step++) {
                    const response = await (this.env.AI as unknown as {
                        run: (model: string, options: {
                            messages: WorkersAIMessage[];
                            tools: typeof TOOLS_SCHEMA;
                            stream: boolean;
                            max_tokens: number;
                        }) => Promise<ReadableStream | { response?: string; tool_calls?: WorkersAIToolCall[] }>;
                    }).run("@cf/meta/llama-3.1-8b-instruct", {
                        messages: stepMessages,
                        tools: TOOLS_SCHEMA,
                        stream: false,
                        max_tokens: 512,
                    });

                    const result = response as { response?: string; tool_calls?: WorkersAIToolCall[] };
                    pendingToolCalls = result.tool_calls ?? [];

                    if (pendingToolCalls.length > 0) {
                        stepMessages.push({
                            role: "assistant",
                            content: "",
                            tool_calls: pendingToolCalls,
                        });

                        for (const tc of pendingToolCalls) {
                            const toolName = tc.function.name;

                            if (calledToolNames.size >= 1) {
                                pendingToolCalls = [];
                                break;
                            }

                            if (calledToolNames.has(toolName)) {
                                // Llama 3 amnesia: it's trying to call the same tool again. Force break.
                                pendingToolCalls = [];
                                stepMessages.push({
                                    role: "user",
                                    content: `System Error: The AI attempted to call the ${toolName} tool repeatedly. Aborting to prevent infinite loop.`
                                });
                                break;
                            }
                            calledToolNames.add(toolName);

                            let inputArgs: Record<string, unknown> = {};
                            try {
                                inputArgs = JSON.parse(tc.function.arguments || "{}");
                            } catch { /* ignore parse error */ }

                            const toolCallId = tc.id || `call_${Math.random().toString(36).slice(2, 9)}`;

                            writer.write({
                                type: "tool-input-available",
                                toolCallId,
                                toolName,
                                input: inputArgs,
                            });

                            if (toolName === "getUserInfo") {
                                writer.write({
                                    type: "tool-output-available",
                                    toolCallId,
                                    output: { pending: "Waiting for browser..." },
                                });
                                return;
                            }

                            if (toolName === "setReminder") {
                                const schema = z.object({
                                    message: z.string(),
                                    delaySeconds: z.number(),
                                });
                                const parsed = schema.safeParse(inputArgs);
                                if (parsed.success) {
                                    const result = await this._scheduleReminder(
                                        parsed.data.message,
                                        parsed.data.delaySeconds
                                    );
                                    writer.write({
                                        type: "tool-output-available",
                                        toolCallId,
                                        output: result,
                                    });
                                    stepMessages.push({
                                        role: "tool",
                                        tool_call_id: toolCallId,
                                        name: toolName,
                                        content: JSON.stringify(result),
                                    });
                                    toolResults.push({ toolCallId, toolName, result });
                                }
                                continue;
                            }

                            if (toolName === "searchWeb") {

                                // 🔒 GUARD: Only allow search if user explicitly requests live/current info
                                const lastUserMessage = historyMessages
                                    .filter(m => m.role === "user")
                                    .slice(-1)[0]?.content?.toLowerCase() ?? "";

                                const triggerWords = ["search", "latest", "current", "today", "news", "real-time"];

                                const shouldSearch = triggerWords.some(word =>
                                    lastUserMessage.includes(word)
                                );

                                if (!shouldSearch) {
                                    console.warn("Blocked unnecessary searchWeb call.");
                                    continue; // 🚫 skip execution
                                }

                                const query = (inputArgs.query as string) ?? "";
                                const searchResult = await executeSearchWeb(query);

                                writer.write({
                                    type: "tool-output-available",
                                    toolCallId,
                                    output: JSON.parse(searchResult),
                                });

                                stepMessages.push({
                                    role: "tool",
                                    tool_call_id: toolCallId,
                                    name: toolName,
                                    content: searchResult,
                                });

                                toolResults.push({
                                    toolCallId,
                                    toolName,
                                    result: JSON.parse(searchResult),
                                });
                            }
                        }

                        if (pendingToolCalls.length > 1) {
                            // We forcefully broke out of the tool loop due to amnesia
                            pendingToolCalls = [pendingToolCalls[0]];
                        }

                        // Continue to next pass (up to 2) so AI can see the tool results
                        continue;
                    } else {
                        assistantText = result.response ?? "";
                        const msgId = `msg-${Date.now()}`;
                        writer.write({
                            type: "text-start",
                            id: msgId,
                        });
                        writer.write({
                            type: "text-delta",
                            delta: assistantText,
                            id: msgId,
                        });
                        writer.write({
                            type: "text-end",
                            id: msgId,
                        });
                        break;
                    }
                }

                this._isGenerating = false;
            },
            onFinish: ({ messages }) => {
                this._isGenerating = false;
                this.saveMessages(messages);
            },
            onError: (error) => {
                this._isGenerating = false;
                console.error("Stream error in ChatAgent:", error);
                return "An error occurred during AI generation.";
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
