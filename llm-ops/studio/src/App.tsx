import { useEffect, useMemo, useState } from "react";
import { Loader2, Send } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

type Health = {
  status: string;
  model_status: "idle" | "loading" | "ready" | "error";
  model_name: string;
  run_dir: string | null;
  using_adapter: boolean;
  error: string | null;
};

type TemplateSample = {
  model_name: string;
  source: string;
  messages: { role: string; content: string }[];
  rendered: string;
  generation_prefix: string;
  notes: string;
};

type Turn = { role: "user" | "assistant"; content: string; prompt?: string };

type Example = { label: string; question: string };

const FALLBACK_PROMPTS: Example[] = [
  {
    label: "Pune cutoff",
    question: "What is the cutoff for same-day dispatch from the Pune hub?",
  },
  { label: "Damage claims", question: "How long do I have to file a damage claim?" },
  { label: "Drone delivery", question: "Does Harborline offer drone delivery?" },
  { label: "COD remittance", question: "When is COD cash remitted to the shipper?" },
];

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const [template, setTemplate] = useState<TemplateSample | null>(null);
  const [prompts, setPrompts] = useState<Example[]>(FALLBACK_PROMPTS);
  const [draft, setDraft] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPrompt, setShowPrompt] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function pull() {
      try {
        const response = await fetch("/api/health");
        if (!response.ok) throw new Error(`Health returned ${response.status}`);
        const body = (await response.json()) as Health;
        if (cancelled) return;
        setHealth(body);
        setOffline(false);
      } catch {
        if (!cancelled) setOffline(true);
      }
    }
    void pull();
    const timer = window.setInterval(pull, 4000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (offline) return;
    void fetch("/api/template")
      .then((response) => response.json())
      .then((body: TemplateSample) => setTemplate(body))
      .catch(() => setTemplate(null));
    void fetch("/api/examples")
      .then((response) => response.json())
      .then((body: { prompts: Example[] }) => {
        if (body.prompts?.length && body.prompts[0].question) setPrompts(body.prompts);
      })
      .catch(() => undefined);
  }, [offline]);

  const statusLabel = useMemo(() => {
    if (offline) return "API offline";
    if (!health) return "Checking API";
    if (health.model_status === "ready") return health.using_adapter ? "Adapter loaded" : "Base model loaded";
    if (health.model_status === "loading") return "Loading weights";
    if (health.model_status === "error") return "Model failed to load";
    return "Weights idle";
  }, [health, offline]);

  async function ask(question: string) {
    const text = question.trim();
    if (!text || busy) return;
    setError(null);
    setBusy(true);
    setDraft("");
    const history: Turn[] = [...turns, { role: "user", content: text }];
    setTurns(history);
    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: history.map(({ role, content }) => ({ role, content })),
        }),
      });
      const body = await response.json();
      if (!response.ok) {
        throw new Error(body.detail || `Query failed (${response.status})`);
      }
      setTurns([
        ...history.slice(0, -1),
        { role: "user", content: text, prompt: body.prompt },
        { role: "assistant", content: body.answer },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setBusy(false);
    }
  }

  const lastPrompt = [...turns].reverse().find((turn) => turn.prompt)?.prompt;

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 py-6 sm:px-6 lg:px-8">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-line pb-5">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-copper">Domain SLM</p>
          <h1 className="mt-1 font-serif text-4xl font-medium tracking-tight text-ink sm:text-5xl">
            Query the trained model
          </h1>
          <p className="mt-2 max-w-xl text-sm leading-relaxed text-ink/70">
            SmolLM2-135M-Instruct is the selected model. Every question is wrapped in its ChatML
            template before generation. The sample rows teach a small Harborline Logistics policy.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge>{health?.model_name ?? "HuggingFaceTB/SmolLM2-135M-Instruct"}</Badge>
          <Badge className={offline ? "border-copper text-copper" : "border-moss text-moss"}>{statusLabel}</Badge>
        </div>
      </header>

      {offline && (
        <p className="mt-4 rounded-2xl border border-copper/40 bg-copper/5 px-4 py-3 text-sm text-ink">
          The query API is not running. From the repo root:{" "}
          <code className="font-mono text-xs">dslm serve --config configs/smollm2-135m-instruct.yaml --port 43123</code>
        </p>
      )}

      <main className="mt-6 grid flex-1 gap-6 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <section className="rounded-3xl border border-line bg-card p-5 shadow-sheet">
          <div className="flex items-center justify-between gap-3">
            <h2 className="font-serif text-2xl">Chat template</h2>
            <Badge>{template?.source ?? "preset"}</Badge>
          </div>
          <p className="mt-2 text-sm leading-relaxed text-ink/70">
            {template?.notes ||
              "ChatML. Each turn is <|im_start|>role, then the text, then <|im_end|>. Generation starts after <|im_start|>assistant."}
          </p>
          <p className="mt-4 font-mono text-[11px] uppercase tracking-[0.14em] text-ink/50">
            One training row, rendered
          </p>
          <pre className="mt-2 max-h-[420px] overflow-auto rounded-2xl bg-ink px-4 py-3 font-mono text-[12px] leading-relaxed text-paper">
            {template?.rendered ?? "Loading the selected model's sample…"}
          </pre>
          <p className="mt-4 text-sm leading-relaxed text-ink/70">
            JSONL rows use <span className="font-mono text-xs">messages</span> with roles system, user,
            and assistant. An older <span className="font-mono text-xs">instruction</span> /{" "}
            <span className="font-mono text-xs">output</span> line is accepted too. Loss is computed only
            on the assistant span, which begins at the generation prefix.
          </p>
        </section>

        <section className="flex min-h-[560px] flex-col rounded-3xl border border-line bg-card p-5 shadow-sheet">
          <div className="flex items-center justify-between gap-3">
            <h2 className="font-serif text-2xl">Harborline policy</h2>
            <Button variant="ghost" size="sm" onClick={() => setShowPrompt((value) => !value)}>
              {showPrompt ? "Hide prompt" : "Show prompt"}
            </Button>
          </div>

          <div className="mt-4 flex flex-1 flex-col gap-3">
            {turns.length === 0 && !busy && (
              <div className="rounded-2xl border border-dashed border-line px-4 py-6 text-sm leading-relaxed text-ink/70">
                No questions yet. Ask about hub cutoffs, claims, COD, or something the policy does not
                mention. The first answer loads the weights, so it takes longer than the ones after it.
                The saved sample adapter has memorized the 12 Harborline rows. A two-step smoke run has not.
              </div>
            )}
            {turns.map((turn, index) => (
              <article
                key={`${turn.role}-${index}`}
                className={
                  turn.role === "user"
                    ? "ml-8 rounded-2xl bg-paper px-4 py-3 text-sm"
                    : "mr-8 rounded-2xl bg-moss/10 px-4 py-3 text-sm"
                }
              >
                <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink/50">{turn.role}</p>
                <p className="mt-1 whitespace-pre-wrap leading-relaxed">{turn.content}</p>
              </article>
            ))}
            {busy && (
              <p className="flex items-center gap-2 text-sm text-ink/70">
                <Loader2 className="h-4 w-4 animate-spin" />
                Formatting the chat template and generating.
              </p>
            )}
            {error && <p className="rounded-2xl bg-copper/10 px-4 py-3 text-sm text-copper">{error}</p>}
          </div>

          {showPrompt && lastPrompt && (
            <pre className="mt-3 max-h-40 overflow-auto rounded-2xl bg-ink/95 px-3 py-2 font-mono text-[11px] leading-relaxed text-paper/90">
              {lastPrompt}
            </pre>
          )}

          <div className="mt-4 flex flex-wrap gap-2">
            {prompts.slice(0, 4).map((prompt) => (
              <Button
                key={prompt.question}
                variant="outline"
                size="sm"
                disabled={busy}
                onClick={() => void ask(prompt.question)}
              >
                {prompt.label}
              </Button>
            ))}
          </div>

          <form
            className="mt-3 flex flex-col gap-3 sm:flex-row"
            onSubmit={(event) => {
              event.preventDefault();
              void ask(draft);
            }}
          >
            <Textarea
              value={draft}
              placeholder="Ask a Harborline policy question"
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void ask(draft);
                }
              }}
            />
            <Button type="submit" disabled={busy || !draft.trim()} className="sm:self-end">
              <Send className="h-4 w-4" />
              Ask
            </Button>
          </form>
        </section>
      </main>
    </div>
  );
}
