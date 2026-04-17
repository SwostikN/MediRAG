import { motion } from "motion/react";
import { AlertTriangle, ExternalLink } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export interface ChatMessageSource {
  rank?: number;
  title?: string | null;
  source?: string | null;
  source_url?: string | null;
  authority_tier?: number | null;
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

function SourcesFooter({ sources }: { sources: ChatMessageSource[] }) {
  const visible = sources.filter((s) => (s.title || s.source_url) != null);
  if (visible.length === 0) return null;

  return (
    <div className="mt-4 border-t border-border/60 pt-3">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
        Sources
      </div>
      <ol className="space-y-1.5 text-[13px] leading-5">
        {visible.map((s, idx) => {
          const label = s.title || s.source || "source";
          const key = `${s.source_url ?? label}-${idx}`;
          return (
            <li key={key} className="flex items-start gap-2">
              <span className="mt-[2px] shrink-0 text-muted-foreground font-mono text-[11px]">
                [{idx + 1}]
              </span>
              {s.source_url ? (
                <a
                  href={s.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group inline-flex items-start gap-1 text-foreground/85 hover:text-accent"
                >
                  <span className="underline decoration-border underline-offset-2 group-hover:decoration-accent">
                    {label}
                  </span>
                  <ExternalLink className="mt-[3px] h-3 w-3 shrink-0 opacity-60 group-hover:opacity-100" />
                </a>
              ) : (
                <span className="text-foreground/85">{label}</span>
              )}
              {s.source && s.title && (
                <span className="ml-auto shrink-0 text-[11px] text-muted-foreground/80">
                  {s.source}
                </span>
              )}
            </li>
          );
        })}
      </ol>
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
          text: "Emergency",
        }
      : {
          container:
            "rounded-2xl border-2 border-amber-500 bg-amber-50 p-5 shadow-md dark:bg-amber-950/40 dark:border-amber-400",
          icon: "text-amber-600 dark:text-amber-300",
          label: "text-amber-700 dark:text-amber-200",
          meta: "text-amber-700/70 dark:text-amber-300/70",
          body: "text-amber-950 dark:text-amber-50",
          text: "Urgent care",
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
            <p
              className={`m-0 whitespace-pre-wrap text-[15px] leading-7 ${
                isUser ? "text-accent-foreground" : "text-foreground/92"
              }`}
            >
              {content}
            </p>
          ) : (
            <div className="relative">
              <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent/60 to-transparent" />
              <div className="pt-1">
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
