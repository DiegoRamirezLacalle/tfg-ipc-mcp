import { useEffect, useRef, useState, type FormEvent } from "react";
import { Sparkles, Send, Loader2, Bot, User as UserIcon } from "lucide-react";

export interface ChatSignalState {
  key: string;
  label: string;
  baseline_value: number;
  current_value: number;
  final_effect?: number | null;
}

export interface SimulatorChatContext {
  series_name: string | null;
  series_unit: string | null;
  horizon: number;
  signals: ChatSignalState[];
  baseline: number[];
  counterfactual: number[];
  top_driver_label?: string | null;
  top_driver_contribution?: number | null;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface Props {
  context: SimulatorChatContext;
}

const SUGGESTIONS = [
  "Explain marginal effect in plain words",
  "Why did the forecast change?",
  "What does ECB Deposit Rate do to inflation?",
  "Baseline vs counterfactual — what's the difference?",
];

export function SimulatorChat({ context }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streaming]);

  useEffect(() => () => abortRef.current?.abort(), []);

  async function send(text: string) {
    const userMsg: ChatMessage = { role: "user", content: text };
    const history: ChatMessage[] = [...messages, userMsg];
    setMessages([...history, { role: "assistant", content: "" }]);
    setInput("");
    setStreaming(true);
    setError(null);

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const token = localStorage.getItem("tfg_token");
      const res = await fetch("/api/v1/assistant/simulator/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ messages: history, context }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        const body = await res.text().catch(() => "");
        throw new Error(`${res.status}: ${body.slice(0, 200) || res.statusText}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let acc = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        acc += decoder.decode(value, { stream: true });
        setMessages((prev) => {
          const copy = prev.slice();
          copy[copy.length - 1] = { role: "assistant", content: acc };
          return copy;
        });
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setError((e as Error).message);
        setMessages((prev) => prev.slice(0, -1));
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || streaming) return;
    send(text);
  }

  return (
    <section className="card-tech flex flex-col overflow-hidden">
      <div className="border-b border-border px-4 py-3 flex items-center gap-2 bg-muted/50">
        <Sparkles size={13} className="text-gold" />
        <span className="micro uppercase">Tutor</span>
        <span className="font-mono text-[10px] text-foreground-subtle ml-1">
          ask anything about this simulation
        </span>
        <span className="pill pill-mcp text-[10px] ml-auto">◈ local LLM</span>
      </div>

      <div ref={scrollRef} className="px-4 py-4 flex flex-col gap-3 max-h-[420px] overflow-y-auto">
        {messages.length === 0 && (
          <div className="flex flex-col gap-3 py-4">
            <p className="font-sans text-sm text-foreground-muted">
              I can explain the variables and tell you <span className="text-gold">why</span> the forecast moved.
              Try a question:
            </p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="px-3 py-1.5 rounded border border-border bg-card hover:border-foreground-subtle hover:text-foreground font-mono text-data-sm text-foreground-muted transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <MessageBubble key={i} role={m.role} content={m.content} streaming={streaming && i === messages.length - 1 && m.role === "assistant"} />
        ))}

        {error && (
          <p className="font-mono text-data-sm text-destructive border border-destructive/30 bg-destructive/10 rounded px-3 py-2">
            {error}
          </p>
        )}
      </div>

      <form onSubmit={onSubmit} className="border-t border-border p-3 flex items-center gap-2 bg-card/40">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about a signal, the result, or any concept…"
          disabled={streaming}
          className="flex-1 bg-background border border-border rounded px-3 py-2 font-mono text-data-sm text-foreground placeholder:text-foreground-subtle focus:border-mcp focus:ring-1 focus:ring-mcp/20 outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={streaming || !input.trim()}
          className="flex items-center justify-center gap-1.5 h-9 px-4 rounded bg-mcp hover:bg-mcp/90 text-white font-mono text-data-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {streaming ? <Loader2 size={12} className="animate-spin" /> : <Send size={12} />}
          Send
        </button>
      </form>
    </section>
  );
}

function MessageBubble({ role, content, streaming }: { role: "user" | "assistant"; content: string; streaming: boolean }) {
  const isUser = role === "user";
  return (
    <div className={"flex gap-2 " + (isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <span className="w-6 h-6 rounded bg-mcp/15 border border-mcp/30 flex items-center justify-center shrink-0 mt-0.5">
          <Bot size={12} className="text-mcp" />
        </span>
      )}
      <div
        className={
          "max-w-[78%] rounded px-3 py-2 font-sans text-sm leading-relaxed whitespace-pre-wrap " +
          (isUser
            ? "bg-mcp/15 border border-mcp/30 text-foreground"
            : "bg-card border border-border text-foreground")
        }
      >
        {content || (streaming ? <span className="text-foreground-subtle">…</span> : null)}
        {streaming && content && <span className="inline-block w-2 h-3 ml-0.5 bg-gold align-middle animate-blink" />}
      </div>
      {isUser && (
        <span className="w-6 h-6 rounded bg-muted border border-border flex items-center justify-center shrink-0 mt-0.5">
          <UserIcon size={12} className="text-foreground-muted" />
        </span>
      )}
    </div>
  );
}
