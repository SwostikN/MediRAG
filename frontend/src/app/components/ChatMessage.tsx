import { Fragment, type ReactNode } from "react";
import { motion } from "motion/react";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
  renderMode?: "plain" | "query";
}

function renderInlineFormatting(text: string): ReactNode[] {
  const segments = text.split(/(\*\*.*?\*\*)/g).filter(Boolean);

  return segments.map((segment, index) => {
    if (segment.startsWith("**") && segment.endsWith("**")) {
      return (
        <strong key={index} className="font-semibold text-foreground">
          {segment.slice(2, -2)}
        </strong>
      );
    }

    return <Fragment key={index}>{segment}</Fragment>;
  });
}

function renderFormattedContent(content: string): ReactNode {
  const lines = content.split("\n");
  const blocks: ReactNode[] = [];
  let paragraphLines: string[] = [];
  let listItems: string[] = [];

  const flushParagraph = () => {
    if (paragraphLines.length === 0) {
      return;
    }

    const paragraph = paragraphLines.join(" ").trim();
    if (paragraph) {
      blocks.push(
        <p key={`p-${blocks.length}`} className="leading-7 text-[15px] text-foreground/92">
          {renderInlineFormatting(paragraph)}
        </p>
      );
    }
    paragraphLines = [];
  };

  const flushList = () => {
    if (listItems.length === 0) {
      return;
    }

    blocks.push(
      <div key={`ol-${blocks.length}`} className="space-y-2 text-[15px] leading-6 text-foreground/92">
        {listItems.map((item, index) => (
          <p key={`${index}-${item.slice(0, 20)}`} className="m-0">
            <span className="mr-2 font-semibold text-accent">{index + 1}.</span>
            {renderInlineFormatting(item)}
          </p>
        ))}
      </div>
    );
    listItems = [];
  };

  lines.forEach((rawLine) => {
    const line = rawLine.trim();
    const numberedMatch = line.match(/^\d+\.\s+(.*)$/);

    if (!line) {
      flushParagraph();
      flushList();
      return;
    }

    if (numberedMatch) {
      flushParagraph();
      listItems.push(numberedMatch[1]);
      return;
    }

    flushList();
    paragraphLines.push(line);
  });

  flushParagraph();
  flushList();

  if (blocks.length === 0) {
    return <p className="leading-7 text-[15px] text-foreground/92">{content}</p>;
  }

  return <div className="space-y-4">{blocks}</div>;
}

export function ChatMessage({ role, content, timestamp, renderMode = "plain" }: ChatMessageProps) {
  const isUser = role === "user";
  const shouldFormatQueryAnswer = !isUser && renderMode === "query";

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
              <div className="pt-1">{renderFormattedContent(content)}</div>
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
