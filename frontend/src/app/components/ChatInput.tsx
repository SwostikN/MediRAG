import { Send, Paperclip, Loader2 } from "lucide-react";
import { useState, useRef } from "react";
import type { ChangeEvent, KeyboardEvent } from "react";
import { motion } from "motion/react";

interface ChatInputProps {
  onSend: (message: string) => void;
  onUpload?: (files: File[]) => void;
  isLoading?: boolean;
  placeholder?: string;
  centered?: boolean;
}

export function ChatInput({
  onSend,
  onUpload,
  isLoading,
  placeholder,
  centered = false,
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
        className={`flex items-end gap-3 ${
          centered
            ? "bg-card border-2 border-accent rounded-2xl p-4 shadow-2xl shadow-accent/20"
            : ""
        }`}
        style={
          centered
            ? {
                boxShadow:
                  "0 0 0 1px rgba(0, 191, 165, 0.1), 0 0 40px rgba(0, 191, 165, 0.15), 0 20px 60px rgba(0, 0, 0, 0.3)",
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
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading}
          className={`shrink-0 p-2.5 rounded-lg border transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            centered
              ? "border-accent/30 hover:bg-accent/10 text-accent"
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

      {!centered && (
        <div className="mt-2 text-xs text-muted-foreground">
          Press Enter to send, Shift+Enter for new line
        </div>
      )}
    </div>
  );
}
