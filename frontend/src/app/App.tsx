import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import type { Session } from "@supabase/supabase-js";
import {
  ArrowLeft,
  Sun,
  Moon,
  Menu,
  Search,
  Settings,
  FileText,
  Activity,
  // Stethoscope,
  Plus,
  ChevronDown,
} from "lucide-react";
import { ChatMessage } from "./components/ChatMessage";
import { ChatInput } from "./components/ChatInput";
import { MedicalContextPanel } from "./components/MedicalContextPanel";
import { EmptyState } from "./components/EmptyState";
import { supabase } from "../lib/supabase";

interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  renderMode?: "plain" | "query";
}

type DesignVariant = "classic" | "diagnostic" | "research";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

function getTimestamp() {
  return new Date().toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function App() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const saved = localStorage.getItem("theme");
    return (saved as "light" | "dark") || "light";
  });
  const [variant, setVariant] = useState<DesignVariant>("classic");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false);
  const [showContextPanel, setShowContextPanel] = useState(true);
  const [uploadedDocuments, setUploadedDocuments] = useState<string[]>([]);
  const [statusMessage, setStatusMessage] = useState("No documents indexed yet");
  const [session, setSession] = useState<Session | null>(null);
  const [authReady, setAuthReady] = useState(false);

  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    let isMounted = true;

    const loadSession = async () => {
      const { data, error } = await supabase.auth.getSession();

      if (!isMounted) {
        return;
      }

      if (error) {
        setStatusMessage(`Supabase auth error: ${error.message}`);
      } else {
        setSession(data.session);
      }

      setAuthReady(true);
    };

    void loadSession();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
      setAuthReady(true);
    });

    return () => {
      isMounted = false;
      subscription.unsubscribe();
    };
  }, []);

  const toggleTheme = () => {
    setTheme(theme === "light" ? "dark" : "light");
  };

  const authLabel = !authReady
    ? "Connecting to Supabase..."
    : session?.user?.email
      ? `Signed in as ${session.user.email}`
      : "Supabase connected";

  const handleUploadPdf = async (files: File[]) => {
    if (files.length === 0) {
      return;
    }

    setIsLoading(true);
    try {
      const uploadedPdfNames: string[] = [];

      for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);

        const uploadResponse = await fetch(`${API_BASE_URL}/upload_pdf`, {
          method: "POST",
          body: formData,
        });

        const uploadPayload = await uploadResponse.json();

        if (!uploadResponse.ok) {
          throw new Error(uploadPayload.detail || "Failed to upload PDF");
        }

        uploadedPdfNames.push(file.name);

        // Save metadata to Supabase `document` table (if exists)
        try {
          const { data: docData, error: docError } = await supabase
            .from("document")
            .insert([
              {
                user_id: session?.user?.id ?? null,
                title: file.name,
                upload_date: new Date().toISOString(),
              },
            ])
            .select()
            .single();

          if (docError) {
            console.warn("Supabase insert document error:", docError.message || docError);
          } else {
            console.debug("Inserted document row:", docData);
          }
        } catch (e) {
          console.warn("Supabase document insert failed", e);
        }
      }

      setUploadedDocuments((prev) => {
        const merged = new Set([...prev, ...uploadedPdfNames]);
        return Array.from(merged);
      });
      setStatusMessage(`Indexed ${uploadedPdfNames.length} PDF file(s)`);
      const uploadedLabel =
        uploadedPdfNames.length === 1 ? uploadedPdfNames[0] : `${uploadedPdfNames.length} documents`;

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: "assistant",
          content: `${uploadedLabel} uploaded successfully. Your document is ready, and you can now ask questions about its contents.`,
          timestamp: getTimestamp(),
          renderMode: "plain",
        },
      ]);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Something went wrong while uploading the PDF";

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: "assistant",
          content: `Error: ${message}`,
          timestamp: getTimestamp(),
          renderMode: "plain",
        },
      ]);
      setStatusMessage("Upload failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (content: string) => {
    const trimmedContent = content.trim();
    const timestamp = getTimestamp();

    if (trimmedContent) {
      const userMessage: Message = {
        id: Date.now(),
        role: "user",
        content: trimmedContent,
        timestamp,
        renderMode: "plain",
      };

      setMessages((prev) => [...prev, userMessage]);
    }

    setIsLoading(true);

    // Persist the query to Supabase `query` table (non-blocking) and capture its id
    let insertedQueryId: number | null = null;
    try {
      const { data: insertedQuery, error: insertQueryError } = await supabase
        .from("query")
        .insert([
          {
            user_id: session?.user?.id ?? null,
            query_text: trimmedContent,
            timestamp: new Date().toISOString(),
          },
        ])
        .select()
        .single();

      if (insertQueryError) {
        console.warn("Supabase insert query error:", insertQueryError.message || insertQueryError);
      } else if (insertedQuery) {
        insertedQueryId = (insertedQuery.query_id as number) ?? (insertedQuery.id as number) ?? null;
        console.debug("Inserted query row id:", insertedQueryId);
      }
    } catch (e) {
      console.warn("Supabase query insert failed", e);
    }

    try {
      if (!trimmedContent) {
        return;
      }

      if (uploadedDocuments.length === 0) {
        throw new Error("Upload a PDF first so the backend can index it before querying.");
      }

      const queryResponse = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: trimmedContent }),
      });

      const queryPayload = await queryResponse.json();

      if (!queryResponse.ok) {
        throw new Error(queryPayload.detail || "Failed to query document");
      }

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 2,
          role: "assistant",
          content: queryPayload.answer,
          timestamp: getTimestamp(),
          renderMode: "query",
        },
      ]);
      setStatusMessage("Backend response received");
      // Persist response to Supabase `response` table (non-blocking)
      try {
        const { error: respError } = await supabase.from("response").insert([
          {
            query_id: insertedQueryId,
            answer: queryPayload.answer,
            confidence_score: queryPayload.confidence ?? null,
            freshness_score: queryPayload.freshness ?? null,
          },
        ]);

        if (respError) {
          console.warn("Supabase insert response error:", respError.message || respError);
        }
      } catch (e) {
        console.warn("Supabase response insert failed", e);
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Something went wrong while contacting the backend";

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 3,
          role: "assistant",
          content: `Error: ${message}`,
          timestamp: getTimestamp(),
          renderMode: "plain",
        },
      ]);
      setStatusMessage("Request failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handlePromptClick = (prompt: string) => {
    void handleSendMessage(prompt);
  };

  const contextMetrics: Array<{ label: string; value: string | number; change: string; status: "good" | "warning" }> = [
    {
      label: "Query Status",
      value: isLoading ? "Running" : "Ready",
      change: statusMessage,
      status: isLoading ? "warning" : "good",
    },
    {
      label: "Messages",
      value: messages.length,
      change: `${messages.filter((message) => message.role === "assistant").length} responses`,
      status: "good",
    },
    {
      label: "Indexed PDFs",
      value: uploadedDocuments.length,
      change: uploadedDocuments.length > 0 ? "Available for retrieval" : "Upload a PDF to begin",
      status: uploadedDocuments.length > 0 ? "good" : "warning",
    },
  ];

  const ragStatus = {
    vectorCount: uploadedDocuments.length,
    lastUpdated: statusMessage,
    quality: uploadedDocuments.length > 0 ? 0.96 : 0.0,
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-background overflow-hidden">
      <div
        className="fixed inset-0 opacity-[0.015] pointer-events-none"
        style={{
          backgroundImage: `radial-gradient(circle, ${
            theme === "dark" ? "#ffffff" : "#000000"
          } 1px, transparent 1px)`,
          backgroundSize: "24px 24px",
        }}
      />

      <motion.header
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="relative z-10 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80"
      >
        <div className="flex items-center justify-between pl-3 pr-6 py-4">
          <div className="flex items-center gap-4">
            {messages.length > 0 && (
              <button
                onClick={() => setMessages([])}
                className="p-2 hover:bg-muted rounded-lg transition-colors"
                aria-label="Back to home"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
            )}

            <button
              onClick={() => setShowSidebar(!showSidebar)}
              className="lg:hidden p-2 hover:bg-muted rounded-lg transition-colors"
            >
              <Menu className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-20">
              <img
                src="/Documed_Logo.png"
                alt="DocuMed AI Logo"
                className="w-40 h-16 rounded-lg object-contain"
                style={{ transform: 'scale(1.8)', transformOrigin: 'left center' }}
              />
              <div>
                <h1 className="text-lg font-medium">DocuMed AI</h1>
                <p className="text-xs text-muted-foreground font-mono">
                  Medical Chat Interface
                </p>
                <p className="text-xs text-muted-foreground/80">
                  {authLabel}
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-2 bg-muted/30 rounded-lg p-1">
              <button
                onClick={() => setVariant("classic")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  variant === "classic"
                    ? "bg-background shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Classic
              </button>
              <button
                onClick={() => setVariant("diagnostic")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  variant === "diagnostic"
                    ? "bg-background shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Diagnostic
              </button>
              <button
                onClick={() => setVariant("research")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  variant === "research"
                    ? "bg-background shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Research
              </button>
            </div>

            <button className="p-2 hover:bg-muted rounded-lg transition-colors">
              <Search className="w-5 h-5" />
            </button>

            <button
              onClick={toggleTheme}
              className="p-2 hover:bg-muted rounded-lg transition-colors"
            >
              {theme === "light" ? (
                <Moon className="w-5 h-5" />
              ) : (
                <Sun className="w-5 h-5" />
              )}
            </button>

            <button className="p-2 hover:bg-muted rounded-lg transition-colors">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>

        {variant === "diagnostic" && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-border px-6 py-3 overflow-hidden"
          >
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-accent" />
                <span className="text-xs font-mono text-muted-foreground">
                  System Status:
                </span>
                <span className="text-xs font-mono font-medium text-accent">
                  {isLoading ? "Processing" : "Optimal"}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-muted-foreground" />
                <span className="text-xs font-mono text-muted-foreground">
                  Active Sources:
                </span>
                <span className="text-xs font-mono font-medium">
                  {uploadedDocuments.length} indexed PDF(s)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
                <span className="text-xs font-mono text-muted-foreground">
                  {statusMessage}
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </motion.header>

      <div className="flex-1 flex overflow-hidden relative">
        <AnimatePresence>
          {(showSidebar || window.innerWidth >= 1024) && (
            <motion.aside
              initial={{ x: -300, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -300, opacity: 0 }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="w-64 border-r border-border bg-card flex flex-col absolute lg:relative inset-y-0 left-0 z-20"
            >
              <div className="p-4 border-b border-border">
                <button
                  onClick={() => {
                    setMessages([]);
                    setStatusMessage("Session reset");
                  }}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-accent text-accent-foreground rounded-lg hover:opacity-90 transition-opacity"
                >
                  <Plus className="w-4 h-4" />
                  <span className="font-medium">New Session</span>
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-2">
                {messages.length > 0 ? (
                  <>
                    <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3 px-2">
                      Recent Activity
                    </div>

                    {messages
                      .filter((message) => message.role === "user")
                      .slice(-4)
                      .reverse()
                      .map((message) => (
                        <button
                          key={message.id}
                          className="w-full text-left px-3 py-2.5 rounded-lg transition-colors hover:bg-muted/50"
                        >
                          <div className="text-sm font-medium mb-0.5 line-clamp-1">
                            {message.content}
                          </div>
                          <div className="text-xs text-muted-foreground font-mono">
                            {message.timestamp}
                          </div>
                        </button>
                      ))}
                  </>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-center px-4">
                    <FileText className="w-12 h-12 text-muted-foreground/30 mb-4" />
                    <p className="text-sm text-muted-foreground">No sessions yet</p>
                    <p className="text-xs text-muted-foreground/70 mt-1">
                      Upload a PDF and start a conversation
                    </p>
                  </div>
                )}
              </div>
            </motion.aside>
          )}
        </AnimatePresence>

        <div className="flex-1 flex flex-col min-w-0">
          {messages.length === 0 ? (
            <EmptyState
              onPromptClick={handlePromptClick}
              inputElement={
                <ChatInput
                  onSend={(message) => {
                    void handleSendMessage(message);
                  }}
                  onUpload={(files) => {
                    void handleUploadPdf(files);
                  }}
                  isLoading={isLoading}
                  centered={false}
                  docked={true}
                  opaque={true}
                  placeholder={
                    variant === "research"
                      ? "Upload PDFs or query your indexed medical research..."
                      : variant === "diagnostic"
                        ? "Upload a clinical PDF and ask a diagnostic question..."
                        : "Upload a PDF and ask about its contents..."
                  }
                />
              }
            />
          ) : (
            <>
              <div className="relative flex-1 overflow-hidden">
                <div
                  className="pointer-events-none absolute inset-x-0 top-0 h-40 z-0"
                  style={{
                    background:
                      theme === "dark"
                        ? "radial-gradient(circle at top, rgba(0, 191, 165, 0.14), transparent 70%)"
                        : "radial-gradient(circle at top, rgba(0, 191, 165, 0.12), transparent 72%)",
                  }}
                />
                <div className="relative z-10 h-full overflow-y-auto px-6 pt-6 pb-44 space-y-6">
                <AnimatePresence mode="popLayout">
                  {messages.map((message) => (
                    <ChatMessage key={message.id} {...message} />
                  ))}
                </AnimatePresence>

                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex gap-4"
                  >
                    <div className="max-w-[80%]">
                      <div className="bg-card border border-border rounded-xl p-4">
                        <div className="flex items-center gap-2">
                          <div className="flex gap-1">
                            <div
                              className="w-2 h-2 rounded-full bg-accent animate-pulse"
                              style={{ animationDelay: "0ms" }}
                            />
                            <div
                              className="w-2 h-2 rounded-full bg-accent animate-pulse"
                              style={{ animationDelay: "150ms" }}
                            />
                            <div
                              className="w-2 h-2 rounded-full bg-accent animate-pulse"
                              style={{ animationDelay: "300ms" }}
                            />
                          </div>
                          <span className="text-sm text-muted-foreground">
                            Contacting DocuMed AI backend...
                          </span>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
              </div>

              <div className="sticky bottom-0 z-20 px-3 pb-3 sm:px-6 sm:pb-5 bg-background">
                <div className="pointer-events-none absolute inset-x-0 bottom-0 h-32 bg-background" />
                <div className="relative">
                  <ChatInput
                    onSend={(message) => {
                      void handleSendMessage(message);
                    }}
                    onUpload={(files) => {
                      void handleUploadPdf(files);
                    }}
                    isLoading={isLoading}
                    centered={false}
                    docked={true}
                    opaque={true}
                    placeholder={
                      variant === "research"
                        ? "Query your indexed medical literature..."
                        : variant === "diagnostic"
                          ? "Ask a question grounded in the uploaded PDF..."
                          : "Ask about the uploaded document..."
                    }
                  />
                </div>
              </div>
            </>
          )}
        </div>

        {(variant === "diagnostic" || variant === "research") && showContextPanel && (
          <motion.div
            initial={{ x: 300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 300, opacity: 0 }}
            className="w-80 hidden xl:block"
          >
            <MedicalContextPanel
              metrics={contextMetrics}
              activeDocuments={uploadedDocuments}
              ragStatus={ragStatus}
            />
          </motion.div>
        )}

        {(variant === "diagnostic" || variant === "research") && (
          <button
            onClick={() => setShowContextPanel(!showContextPanel)}
            className="hidden xl:flex absolute top-1/2 right-0 -translate-y-1/2 p-2 bg-card border border-border rounded-l-lg hover:bg-muted transition-colors z-10"
            style={{ right: showContextPanel ? "320px" : "0" }}
          >
            <ChevronDown
              className={`w-4 h-4 transition-transform ${
                showContextPanel ? "rotate-90" : "-rotate-90"
              }`}
            />
          </button>
        )}
      </div>
    </div>
  );
}
