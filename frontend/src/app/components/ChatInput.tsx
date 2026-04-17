import { Send, Paperclip, Loader2 } from "lucide-react";
import { useState, useRef } from "react";
import type { ChangeEvent, KeyboardEvent } from "react";
import { motion } from "motion/react";

interface ChatInputProps {
  onSend: (message: string) => void;
  onUpload?: (files: File[]) => void;
  onUploadClick?: () => boolean | void;
  isLoading?: boolean;
  placeholder?: string;
  centered?: boolean;
  docked?: boolean;
  opaque?: boolean;
}

export function ChatInput({
  onSend,
  onUpload,
  onUploadClick,
  isLoading,
  placeholder,
  centered = false,
  docked = false,
  opaque = false,
}: ChatInputProps) {
  const [message, setMessage] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    if (message.trim()) {
      onSend(message.trim());
      setMessage("");
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
      }
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
  };

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      setFiles(selectedFiles);
      onUpload?.(selectedFiles);
      setFiles([]);
      e.target.value = "";
    }
  };

  return (
    <div
      className={
        centered
          ? "w-full max-w-3xl mx-auto px-6"
          : docked
            ? "w-full max-w-4xl mx-auto px-4 sm:px-6"
            : "border-t border-border bg-background p-4"
      }
    >
      {files.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {files.map((file, idx) => (
            <div
              key={idx}
              className="flex items-center gap-2 bg-muted rounded-lg px-3 py-1.5 text-xs"
            >
              <Paperclip className="w-3 h-3" />
              <span className="font-medium">{file.name}</span>
              <button
                onClick={() => setFiles(files.filter((_, i) => i !== idx))}
                className="ml-1 text-muted-foreground hover:text-foreground"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <motion.div
        className={`flex items-center gap-3 ${
          centered
            ? "bg-card border-2 border-accent rounded-2xl p-4 shadow-2xl shadow-accent/20"
            : docked
              ? `rounded-[1.75rem] border p-3.5 shadow-[0_24px_90px_rgba(0,0,0,0.2)] ${
                  opaque
                    ? "border-border bg-card"
                    : "border-border bg-card"
                }`
              : ""
        }`}
        style={
          centered
            ? {
                boxShadow:
                  "0 0 0 1px rgba(0, 191, 165, 0.1), 0 0 40px rgba(0, 191, 165, 0.15), 0 20px 60px rgba(0, 0, 0, 0.3)",
              }
            : docked
              ? {
                  boxShadow:
                    "0 0 0 1px rgba(0, 191, 165, 0.12), 0 20px 50px rgba(15, 23, 42, 0.18), 0 0 60px rgba(0, 191, 165, 0.08)",
                }
            : undefined
        }
        animate={
          centered
            ? {
                boxShadow: [
                  "0 0 0 1px rgba(0, 191, 165, 0.1), 0 0 40px rgba(0, 191, 165, 0.15), 0 20px 60px rgba(0, 0, 0, 0.3)",
                  "0 0 0 1px rgba(0, 191, 165, 0.2), 0 0 50px rgba(0, 191, 165, 0.25), 0 20px 60px rgba(0, 0, 0, 0.3)",
                  "0 0 0 1px rgba(0, 191, 165, 0.1), 0 0 40px rgba(0, 191, 165, 0.15), 0 20px 60px rgba(0, 0, 0, 0.3)",
                ],
              }
            : undefined
        }
        transition={
          centered
            ? {
                duration: 3,
                repeat: Infinity,
                ease: "easeInOut",
              }
            : undefined
        }
      >
        <button
          onClick={() => {
            if (onUploadClick?.() === false) return;
            fileInputRef.current?.click();
          }}
          disabled={isLoading}
          className={`shrink-0 p-2.5 rounded-lg border transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            centered
              ? "border-accent/30 hover:bg-accent/10 text-accent"
              : docked
                ? "border-accent/20 bg-accent/5 text-accent hover:bg-accent/12"
              : "border-border hover:bg-muted"
          }`}
        >
          <Paperclip className={centered ? "w-5 h-5" : "w-5 h-5"} />
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={handleFileSelect}
          accept=".pdf,.txt,.doc,.docx"
        />

        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            placeholder={placeholder || "Ask about medical information..."}
            rows={1}
            className={`w-full resize-none rounded-lg px-4 py-3 pr-12 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed ${
              centered
                ? "border-0 bg-transparent focus:ring-0 text-base placeholder:text-muted-foreground/60"
                : docked
                  ? "border-0 bg-card text-base placeholder:text-muted-foreground/70 focus:ring-0"
                : "border border-border bg-background focus:ring-2 focus:ring-accent"
            }`}
            style={{
              minHeight: centered ? "52px" : "44px",
              maxHeight: "200px",
            }}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !message.trim()}
            className={`absolute right-2 p-2 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed ${
              centered
                ? "bottom-2.5 bg-accent text-accent-foreground hover:scale-105 shadow-lg"
                : docked
                  ? "bottom-2.5 bg-accent text-accent-foreground hover:scale-105 shadow-lg shadow-accent/25"
                : "bottom-2 bg-accent text-accent-foreground hover:opacity-90"
            }`}
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </motion.div>

      {!centered && !docked && (
        <div className="mt-2 text-xs text-muted-foreground">
          Press Enter to send, Shift+Enter for new line
        </div>
      )}
    </div>
  );
}
