import { motion } from "motion/react";
import { AlertTriangle, ExternalLink, Phone } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export interface ChatMessageSource {
  rank?: number;
  title?: string | null;
  source?: string | null;
  source_url?: string | null;
  authority_tier?: number | null;
  publication_date?: string | null;
}

type FreshnessTier = "current" | "recent" | "older" | "unknown";

function classifyFreshness(publication_date: string | null | undefined): {
  tier: FreshnessTier;
  label: string;
} {
  if (!publication_date) {
    return { tier: "unknown", label: "undated" };
  }
  const match = /^(\d{4})/.exec(publication_date);
  if (!match) {
    return { tier: "unknown", label: "undated" };
  }
  const year = parseInt(match[1], 10);
  const currentYear = new Date().getFullYear();
  const age = currentYear - year;
  if (age <= 1) return { tier: "current", label: `${year}` };
  if (age <= 3) return { tier: "recent", label: `${year}` };
  return { tier: "older", label: `${year}` };
}

const FRESHNESS_STYLES: Record<FreshnessTier, string> = {
  current:
    "bg-emerald-100 text-emerald-800 border-emerald-200 dark:bg-emerald-950/50 dark:text-emerald-200 dark:border-emerald-800/60",
  recent:
    "bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-950/50 dark:text-amber-200 dark:border-amber-800/60",
  older:
    "bg-red-100 text-red-800 border-red-200 dark:bg-red-950/50 dark:text-red-200 dark:border-red-800/60",
  unknown:
    "bg-muted text-muted-foreground border-border",
};

export interface ChatMessageMarker {
  name: string;
  value: number;
  unit: string;
  reference_range: string | null;
  status: "low" | "normal" | "high" | "unknown";
}

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
  renderMode?: "plain" | "query";
  redFlag?: {
    ruleId: string;
    category: string;
    urgency: string;
  };
  sources?: ChatMessageSource[];
  markers?: ChatMessageMarker[];
  // When set, render two disambiguation buttons under the message
  // body for an upload the classifier couldn't bucket. The parent
  // owns the click → POST /upload/resolve flow.
  resolveActions?: {
    sessionDocId: string;
    filename: string;
  };
  onResolveUpload?: (
    sessionDocId: string,
    docType: "lab_report" | "research_paper",
  ) => void;
  // Disables the disambiguation buttons after a click so the user
  // can't fire the resolve twice while waiting on the response.
  resolvePending?: boolean;
}

const markdownComponents = {
  h1: ({ children }: { children?: React.ReactNode }) => (
    <h3 className="mt-4 mb-2 text-base font-semibold text-foreground">{children}</h3>
  ),
  h2: ({ children }: { children?: React.ReactNode }) => (
    <h3 className="mt-4 mb-2 text-base font-semibold text-foreground">{children}</h3>
  ),
  h3: ({ children }: { children?: React.ReactNode }) => (
    <h4 className="mt-3 mb-1.5 text-[15px] font-semibold text-foreground">{children}</h4>
  ),
  p: ({ children }: { children?: React.ReactNode }) => (
    <p className="mb-3 last:mb-0 text-[15px] leading-7 text-foreground/92">{children}</p>
  ),
  ul: ({ children }: { children?: React.ReactNode }) => (
    <ul className="mb-3 last:mb-0 ml-5 list-disc space-y-1.5 text-[15px] leading-6 text-foreground/92 marker:text-accent">
      {children}
    </ul>
  ),
  ol: ({ children }: { children?: React.ReactNode }) => (
    <ol className="mb-3 last:mb-0 ml-5 list-decimal space-y-1.5 text-[15px] leading-6 text-foreground/92 marker:text-accent marker:font-semibold">
      {children}
    </ol>
  ),
  li: ({ children }: { children?: React.ReactNode }) => (
    <li className="pl-1 [&>p]:mb-0">{children}</li>
  ),
  strong: ({ children }: { children?: React.ReactNode }) => (
    <strong className="font-semibold text-foreground">{children}</strong>
  ),
  em: ({ children }: { children?: React.ReactNode }) => (
    <em className="italic text-foreground/90">{children}</em>
  ),
  hr: () => <hr className="my-4 border-border/60" />,
  a: ({ href, children }: { href?: string; children?: React.ReactNode }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
    >
      {children}
    </a>
  ),
  code: ({ children }: { children?: React.ReactNode }) => (
    <code className="rounded bg-muted px-1.5 py-0.5 text-[13px] font-mono text-foreground/90">
      {children}
    </code>
  ),
  blockquote: ({ children }: { children?: React.ReactNode }) => (
    <blockquote className="mb-3 border-l-4 border-accent/40 pl-4 italic text-foreground/85">
      {children}
    </blockquote>
  ),
};

// Inline lab-marker table for Stage 4 explainer responses. The table
// renders BEFORE the prose because the patient's own values are the
// most useful thing on screen — the prose explains them.
function MarkersTable({ markers }: { markers: ChatMessageMarker[] }) {
  if (!markers || markers.length === 0) return null;

  const statusBadge = (status: ChatMessageMarker["status"]) => {
    const base =
      "inline-block rounded px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide";
    switch (status) {
      case "high":
        return `${base} bg-red-100 text-red-700 dark:bg-red-950/60 dark:text-red-200`;
      case "low":
        return `${base} bg-amber-100 text-amber-700 dark:bg-amber-950/60 dark:text-amber-200`;
      case "normal":
        return `${base} bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-200`;
      default:
        return `${base} bg-muted text-muted-foreground`;
    }
  };

  return (
    <div className="mb-4 overflow-x-auto rounded-xl border border-border/70 bg-background/40">
      <table className="w-full text-[13px]">
        <thead className="bg-muted/40 text-left text-[11px] uppercase tracking-wider text-muted-foreground">
          <tr>
            <th className="px-3 py-2 font-semibold">Marker</th>
            <th className="px-3 py-2 font-semibold">Value</th>
            <th className="px-3 py-2 font-semibold">Reference</th>
            <th className="px-3 py-2 font-semibold">Status</th>
          </tr>
        </thead>
        <tbody>
          {markers.map((m, idx) => (
            <tr
              key={`${m.name}-${idx}`}
              className="border-t border-border/50 align-middle"
            >
              <td className="px-3 py-2 font-medium text-foreground">{m.name}</td>
              <td className="px-3 py-2 font-mono text-foreground/90">
                {m.value} <span className="text-muted-foreground">{m.unit}</span>
              </td>
              <td className="px-3 py-2 font-mono text-muted-foreground">
                {m.reference_range || "—"}
              </td>
              <td className="px-3 py-2">
                <span className={statusBadge(m.status)}>{m.status}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SourceChip({ source, index }: { source: ChatMessageSource; index: number }) {
  const title = source.title || source.source || "source";
  const domain = source.source || null;
  const freshness = classifyFreshness(source.publication_date);
  const hasUrl = Boolean(source.source_url);

  const Wrapper: React.ElementType = hasUrl ? "a" : "div";
  const wrapperProps = hasUrl
    ? {
        href: source.source_url as string,
        target: "_blank",
        rel: "noopener noreferrer",
      }
    : {};

  return (
    <Wrapper
      {...wrapperProps}
      className={`group inline-flex max-w-full items-center gap-2 rounded-full border border-border bg-card/80 px-3 py-1.5 text-[12px] leading-5 transition-colors ${
        hasUrl ? "hover:border-accent/60 hover:bg-accent/5" : ""
      }`}
    >
      <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
        [{index + 1}]
      </span>
      {domain && (
        <span className="shrink-0 font-semibold uppercase tracking-wide text-[10px] text-accent">
          {domain}
        </span>
      )}
      <span className="truncate text-foreground/90" title={title}>
        {title}
      </span>
      <span
        className={`shrink-0 rounded-full border px-2 py-[1px] font-mono text-[10px] ${FRESHNESS_STYLES[freshness.tier]}`}
        title={
          freshness.tier === "current"
            ? "Recent guidance"
            : freshness.tier === "recent"
              ? "A few years old — still likely current"
              : freshness.tier === "older"
                ? "Older source — check for newer guidance"
                : "Publication date unknown"
        }
      >
        {freshness.label}
      </span>
      {hasUrl && (
        <ExternalLink className="h-3 w-3 shrink-0 opacity-60 group-hover:opacity-100" />
      )}
    </Wrapper>
  );
}

function SourcesFooter({ sources }: { sources: ChatMessageSource[] }) {
  const visible = sources.filter((s) => (s.title || s.source_url) != null);
  if (visible.length === 0) return null;

  return (
    <div className="mt-4 border-t border-border/60 pt-3">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
        Sources
      </div>
      <div className="flex flex-wrap gap-2">
        {visible.map((s, idx) => (
          <SourceChip key={`${s.source_url ?? s.title ?? "src"}-${idx}`} source={s} index={idx} />
        ))}
      </div>
    </div>
  );
}

export function ChatMessage({
  role,
  content,
  timestamp,
  renderMode = "plain",
  redFlag,
  sources,
  markers,
  resolveActions,
  onResolveUpload,
  resolvePending,
}: ChatMessageProps) {
  const isUser = role === "user";
  const shouldFormatQueryAnswer = !isUser && renderMode === "query";
  const isRedFlag = !isUser && redFlag !== undefined;

  if (isRedFlag) {
    const isEmergency = redFlag?.urgency === "emergency";
    const banner = isEmergency
      ? {
          container:
            "rounded-2xl border-2 border-red-500 bg-red-50 p-5 shadow-md dark:bg-red-950/40 dark:border-red-400",
          icon: "text-red-600 dark:text-red-300",
          label: "text-red-700 dark:text-red-200",
          meta: "text-red-700/70 dark:text-red-300/70",
          body: "text-red-950 dark:text-red-50",
          text: "Emergency — call 102 now",
          cta: "bg-red-600 hover:bg-red-700 text-white dark:bg-red-500 dark:hover:bg-red-400",
          disclaimer: "text-red-900/75 dark:text-red-200/75",
        }
      : {
          container:
            "rounded-2xl border-2 border-amber-500 bg-amber-50 p-5 shadow-md dark:bg-amber-950/40 dark:border-amber-400",
          icon: "text-amber-600 dark:text-amber-300",
          label: "text-amber-700 dark:text-amber-200",
          meta: "text-amber-700/70 dark:text-amber-300/70",
          body: "text-amber-950 dark:text-amber-50",
          text: "Urgent care needed",
          cta: "bg-amber-600 hover:bg-amber-700 text-white dark:bg-amber-500 dark:hover:bg-amber-400",
          disclaimer: "text-amber-900/75 dark:text-amber-200/75",
        };
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className="flex gap-4 justify-start"
      >
        <div className="max-w-[82%] w-full">
          <div className={banner.container}>
            <div className="mb-3 flex items-center gap-2">
              <AlertTriangle className={`h-5 w-5 ${banner.icon}`} aria-hidden="true" />
              <span className={`text-sm font-bold uppercase tracking-wide ${banner.label}`}>
                {banner.text}
              </span>
              <span className={`ml-auto font-mono text-[11px] ${banner.meta}`}>
                {redFlag?.category} · {redFlag?.ruleId}
              </span>
            </div>
            <div className={`text-[15px] leading-7 whitespace-pre-wrap ${banner.body}`}>
              {content}
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-3">
              <a
                href="tel:102"
                className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold shadow-sm transition-colors ${banner.cta}`}
              >
                <Phone className="h-4 w-4" aria-hidden="true" />
                Call 102 (ambulance)
              </a>
              <span className={`text-[11px] ${banner.disclaimer}`}>
                This is not a diagnosis. MediRAG is a navigator — seek in-person care.
              </span>
            </div>
          </div>
          {timestamp && (
            <div className="mt-2 px-1 text-left">
              <span className="text-xs text-muted-foreground font-mono">{timestamp}</span>
            </div>
          )}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={`flex gap-4 ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div className={`max-w-[82%] ${isUser ? "order-2" : ""}`}>
        <div
          className={`rounded-2xl p-5 shadow-sm ${
            isUser
              ? "bg-accent text-accent-foreground"
              : "border border-border bg-card/95 backdrop-blur"
          }`}
        >
          {!shouldFormatQueryAnswer ? (
            <>
              <p
                className={`m-0 whitespace-pre-wrap text-[15px] leading-7 ${
                  isUser ? "text-accent-foreground" : "text-foreground/92"
                }`}
              >
                {content}
              </p>
              {resolveActions && onResolveUpload && (
                <div className="mt-3 flex flex-wrap gap-2">
                  <button
                    type="button"
                    disabled={resolvePending}
                    onClick={() =>
                      onResolveUpload(resolveActions.sessionDocId, "lab_report")
                    }
                    className="rounded-full border border-emerald-300 bg-emerald-50 px-3 py-1.5 text-xs font-semibold text-emerald-700 transition hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-60 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-200 dark:hover:bg-emerald-950/60"
                  >
                    Treat as lab report
                  </button>
                  <button
                    type="button"
                    disabled={resolvePending}
                    onClick={() =>
                      onResolveUpload(
                        resolveActions.sessionDocId,
                        "research_paper",
                      )
                    }
                    className="rounded-full border border-sky-300 bg-sky-50 px-3 py-1.5 text-xs font-semibold text-sky-700 transition hover:bg-sky-100 disabled:cursor-not-allowed disabled:opacity-60 dark:border-sky-800 dark:bg-sky-950/40 dark:text-sky-200 dark:hover:bg-sky-950/60"
                  >
                    Treat as research paper
                  </button>
                </div>
              )}
            </>
          ) : (
            <div className="relative">
              <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent/60 to-transparent" />
              <div className="pt-1">
                {markers && markers.length > 0 && <MarkersTable markers={markers} />}
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                  {content}
                </ReactMarkdown>
                {sources && sources.length > 0 && <SourcesFooter sources={sources} />}
              </div>
            </div>
          )}
        </div>

        {timestamp && (
          <div className={`mt-2 px-1 ${isUser ? "text-right" : "text-left"}`}>
            <span className="text-xs text-muted-foreground font-mono">{timestamp}</span>
          </div>
        )}
      </div>
    </motion.div>
  );
}
