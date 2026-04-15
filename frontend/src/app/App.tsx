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
  Plus,
  ChevronDown,
  LogIn,
  LogOut,
} from "lucide-react";
import { ChatMessage } from "./components/ChatMessage";
import { ChatInput } from "./components/ChatInput";
import { MedicalContextPanel } from "./components/MedicalContextPanel";
import { EmptyState } from "./components/EmptyState";
import { AuthScreen } from "./components/AuthScreen";
import { supabase } from "../lib/supabase";

interface Message {
  id: number | string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  renderMode?: "plain" | "query";
}

interface ChatSessionRecord {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  last_message_preview: string | null;
}

interface ChatMessageRecord {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  render_mode: "plain" | "query" | null;
}

type DesignVariant = "classic" | "diagnostic" | "research";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

function getTimestamp(date = new Date()) {
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatStoredTimestamp(value: string) {
  return getTimestamp(new Date(value));
}

function generateSessionTitle(seed: string) {
  const trimmed = seed.trim().replace(/\s+/g, " ");
  if (!trimmed) {
    return "Untitled session";
  }

  return trimmed.length > 48 ? `${trimmed.slice(0, 45)}...` : trimmed;
}

function toUiMessages(rows: ChatMessageRecord[]): Message[] {
  return rows.map((row) => ({
    id: row.id,
    role: row.role,
    content: row.content,
    timestamp: formatStoredTimestamp(row.created_at),
    renderMode: row.render_mode ?? "plain",
  }));
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
  const [showAuthScreen, setShowAuthScreen] = useState(false);
  const [conversationHistory, setConversationHistory] = useState<ChatSessionRecord[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [historyAvailable, setHistoryAvailable] = useState(true);

  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    localStorage.setItem("theme", theme);
  }, [theme]);

  const resetLocalConversation = (nextStatus = "No documents indexed yet") => {
    setMessages([]);
    setUploadedDocuments([]);
    setCurrentSessionId(null);
    setStatusMessage(nextStatus);
  };

  const syncUserProfile = async (activeSession: Session) => {
    const metadata = activeSession.user.user_metadata ?? {};
    const fullName =
      typeof metadata.full_name === "string" && metadata.full_name.trim()
        ? metadata.full_name.trim()
        : null;

    const { error } = await supabase.from("user_profiles").upsert(
      [
        {
          id: activeSession.user.id,
          email: activeSession.user.email,
          full_name: fullName,
          last_login_at: new Date().toISOString(),
        },
      ],
      { onConflict: "id" },
    );

    if (error) {
      throw error;
    }
  };

  const loadConversationMessages = async (chatSessionId: string) => {
    const { data, error } = await supabase
      .from("chat_messages")
      .select("id, session_id, role, content, created_at, render_mode")
      .eq("session_id", chatSessionId)
      .order("created_at", { ascending: true });

    if (error) {
      throw error;
    }

    setMessages(toUiMessages((data ?? []) as ChatMessageRecord[]));
    setCurrentSessionId(chatSessionId);
    setUploadedDocuments([]);
    setStatusMessage("Session restored. Re-upload the PDF if you want to continue querying.");
  };

  const loadConversationHistory = async (userId: string) => {
    const { data, error } = await supabase
      .from("chat_sessions")
      .select("id, title, created_at, updated_at, last_message_preview")
      .eq("user_id", userId)
      .order("updated_at", { ascending: false });

    if (error) {
      throw error;
    }

    const history = (data ?? []) as ChatSessionRecord[];
    setConversationHistory(history);

    if (history.length === 0) {
      resetLocalConversation("No saved sessions yet. Upload a PDF to begin.");
      return;
    }

    resetLocalConversation("Signed in successfully.");
  };

  useEffect(() => {
    let isMounted = true;

    const loadSession = async () => {
      const { data, error } = await supabase.auth.getSession();

      if (!isMounted) {
        return;
      }

      if (error) {
        setStatusMessage(`Supabase auth error: ${error.message}`);
        setAuthReady(true);
        return;
      }

      const activeSession = data.session;
      setSession(activeSession);
      setAuthReady(true);

      if (!activeSession) {
        setConversationHistory([]);
        return;
      }

      try {
        await syncUserProfile(activeSession);
        setHistoryAvailable(true);
        await loadConversationHistory(activeSession.user.id);
      } catch (historyError) {
        console.warn("Failed to load chat history", historyError);
        setHistoryAvailable(false);
        setConversationHistory([]);
        setStatusMessage("Signed in. Supabase history tables are unavailable, so chat is running without saved history.");
      }
    };

    void loadSession();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
      setAuthReady(true);

      if (!nextSession) {
        setConversationHistory([]);
        resetLocalConversation("Signed out. Local session cleared.");
        return;
      }

      void (async () => {
        try {
          await syncUserProfile(nextSession);
          setHistoryAvailable(true);
          await loadConversationHistory(nextSession.user.id);
          setShowAuthScreen(false);
        } catch (historyError) {
          console.warn("Failed to refresh chat history", historyError);
          setHistoryAvailable(false);
          setConversationHistory([]);
          setStatusMessage("Signed in. Supabase history tables are unavailable, so chat is running without saved history.");
        }
      })();
    });

    return () => {
      isMounted = false;
      subscription.unsubscribe();
    };
  }, []);

  const toggleTheme = () => {
    setTheme(theme === "light" ? "dark" : "light");
  };

  const upsertConversationHistory = (entry: ChatSessionRecord) => {
    setConversationHistory((prev) =>
      [entry, ...prev.filter((sessionItem) => sessionItem.id !== entry.id)].sort(
        (left, right) =>
          new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime(),
      ),
    );
  };

  const createConversation = async (seed: string) => {
    if (!session?.user?.id) {
      return null;
    }

    const now = new Date().toISOString();
    const payload = {
      user_id: session.user.id,
      title: generateSessionTitle(seed),
      last_message_preview: null,
      created_at: now,
      updated_at: now,
    };

    const { data, error } = await supabase
      .from("chat_sessions")
      .insert([payload])
      .select("id, title, created_at, updated_at, last_message_preview")
      .single();

    if (error) {
      throw error;
    }

    const created = data as ChatSessionRecord;
    upsertConversationHistory(created);
    setCurrentSessionId(created.id);
    return created.id;
  };

  const persistConversationMessage = async (
    chatSessionId: string,
    message: Pick<Message, "role" | "content" | "renderMode">,
  ) => {
    const now = new Date().toISOString();
    const preview = message.content.slice(0, 140);

    const { error: insertError } = await supabase.from("chat_messages").insert([
      {
        session_id: chatSessionId,
        role: message.role,
        content: message.content,
        render_mode: message.renderMode ?? "plain",
      },
    ]);

    if (insertError) {
      console.warn("Supabase message insert error:", insertError.message || insertError);
      return;
    }

    const { data: updatedSession, error: updateError } = await supabase
      .from("chat_sessions")
      .update({
        updated_at: now,
        last_message_preview: preview,
      })
      .eq("id", chatSessionId)
      .select("id, title, created_at, updated_at, last_message_preview")
      .single();

    if (updateError) {
      console.warn("Supabase session update error:", updateError.message || updateError);
      return;
    }

    upsertConversationHistory(updatedSession as ChatSessionRecord);
  };

  const ensureConversationId = async (seed: string) => {
    if (currentSessionId) {
      return currentSessionId;
    }

    if (!session?.user?.id || !historyAvailable) {
      return null;
    }

    try {
      return await createConversation(seed);
    } catch (error) {
      console.warn("Unable to create chat session in Supabase", error);
      setHistoryAvailable(false);
      setStatusMessage("Supabase history tables are unavailable. Upload and chat still work, but this session will not be saved.");
      return null;
    }
  };

  const buildRequestHeaders = (includeJson = false) => {
    const headers: Record<string, string> = {};

    if (includeJson) {
      headers["Content-Type"] = "application/json";
    }

    return headers;
  };

  const handleUploadPdf = async (files: File[]) => {
    if (files.length === 0) {
      return;
    }

    setIsLoading(true);
    try {
      const uploadedPdfNames: string[] = [];
      const chatSessionId = await ensureConversationId(files[0].name);

      for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);

        const uploadResponse = await fetch(`${API_BASE_URL}/upload_pdf`, {
          method: "POST",
          headers: buildRequestHeaders(false),
          body: formData,
        });

        const uploadPayload = await uploadResponse.json();

        if (!uploadResponse.ok) {
          throw new Error(uploadPayload.detail || "Failed to upload PDF");
        }

        uploadedPdfNames.push(file.name);
      }

      setUploadedDocuments((prev) => Array.from(new Set([...prev, ...uploadedPdfNames])));
      setStatusMessage(
        session?.user?.id
          ? `Indexed ${uploadedPdfNames.length} PDF file(s) and linked them to your account.`
          : `Indexed ${uploadedPdfNames.length} PDF file(s) in local mode.`,
      );

      const uploadedLabel =
        uploadedPdfNames.length === 1 ? uploadedPdfNames[0] : `${uploadedPdfNames.length} documents`;

      const assistantMessage: Message = {
        id: Date.now(),
        role: "assistant",
        content: `${uploadedLabel} uploaded successfully. Your document is ready, and you can now ask questions about its contents.`,
        timestamp: getTimestamp(),
        renderMode: "plain",
      };

      setMessages((prev) => [...prev, assistantMessage]);

      if (chatSessionId) {
        void persistConversationMessage(chatSessionId, assistantMessage);
      } else if (session?.user?.id) {
        setStatusMessage("Document uploaded. Supabase history tables are unavailable, so this session is running without saved history.");
      }
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

    if (!trimmedContent) {
      return;
    }

    const chatSessionId = await ensureConversationId(trimmedContent);

    const userMessage: Message = {
      id: Date.now(),
      role: "user",
      content: trimmedContent,
      timestamp,
      renderMode: "plain",
    };

    setMessages((prev) => [...prev, userMessage]);

    if (chatSessionId) {
      void persistConversationMessage(chatSessionId, userMessage);
    } else if (!session?.user?.id) {
      setStatusMessage("Running in local mode. Sign in to keep per-user history.");
    }

    setIsLoading(true);

    try {
      if (uploadedDocuments.length === 0) {
        throw new Error("Upload a PDF first so the backend can index it before querying.");
      }

      const queryResponse = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: buildRequestHeaders(true),
        body: JSON.stringify({ question: trimmedContent }),
      });

      const queryPayload = await queryResponse.json();

      if (!queryResponse.ok) {
        throw new Error(queryPayload.detail || "Failed to query document");
      }

      const assistantMessage: Message = {
        id: Date.now() + 2,
        role: "assistant",
        content: queryPayload.answer,
        timestamp: getTimestamp(),
        renderMode: "query",
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setStatusMessage("Backend response received");

      if (chatSessionId) {
        void persistConversationMessage(chatSessionId, assistantMessage);
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Something went wrong while contacting the backend";

      const errorAssistantMessage: Message = {
        id: Date.now() + 3,
        role: "assistant",
        content: `Error: ${message}`,
        timestamp: getTimestamp(),
        renderMode: "plain",
      };

      setMessages((prev) => [...prev, errorAssistantMessage]);
      setStatusMessage("Request failed");

      if (chatSessionId) {
        void persistConversationMessage(chatSessionId, errorAssistantMessage);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handlePromptClick = (prompt: string) => {
    void handleSendMessage(prompt);
  };

  const handleOpenConversation = (chatSessionId: string) => {
    void loadConversationMessages(chatSessionId);
  };

  const handleStartNewSession = () => {
    resetLocalConversation(
      session?.user?.id
        ? "Ready for a new saved session."
        : "New local session ready. Sign in to save future history.",
    );
  };

  const handleSignIn = async (email: string, password: string) => {
    const { error } = await supabase.auth.signInWithPassword({ email, password });

    if (error) {
      throw error;
    }

    setStatusMessage("Signed in successfully.");
    setShowAuthScreen(false);
  };

  const handleSignUp = async (email: string, password: string, fullName: string) => {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: {
          full_name: fullName,
        },
      },
    });

    if (error) {
      throw error;
    }

    if (data.session) {
      setStatusMessage("Account created and signed in.");
      setShowAuthScreen(false);
      return;
    }

    setStatusMessage("Account created. Check your email to confirm, then sign in.");
  };

  const handleSignOut = async () => {
    const { error } = await supabase.auth.signOut();

    if (error) {
      console.warn("Supabase sign out error:", error.message || error);
      return;
    }

    setShowAuthScreen(false);
  };

  const authLabel = !authReady
    ? "Connecting to Supabase..."
    : session?.user?.email
      ? `Signed in as ${session.user.email}`
      : "";

  const contextMetrics: Array<{
    label: string;
    value: string | number;
    change: string;
    status: "good" | "warning";
  }> = [
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

  const isDesktop = typeof window !== "undefined" ? window.innerWidth >= 1024 : true;

  if (showAuthScreen) {
    return (
      <AuthScreen
        onBack={() => setShowAuthScreen(false)}
        onSignIn={handleSignIn}
        onSignUp={handleSignUp}
        statusMessage={statusMessage}
      />
    );
  }

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
                onClick={handleStartNewSession}
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
                style={{ transform: "scale(1.8)", transformOrigin: "left center" }}
              />
              <div>
                <h1 className="text-lg font-medium">DocuMed AI</h1>
                <p className="text-xs text-muted-foreground font-mono">Medical Chat Interface</p>
                <p className="text-xs text-muted-foreground/80">{authLabel}</p>
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

            {!session ? (
              <button
                onClick={() => setShowAuthScreen(true)}
                className="hidden md:inline-flex items-center gap-2 rounded-lg border border-accent/20 bg-accent/10 px-3 py-2 text-sm text-accent transition-colors hover:bg-accent/15"
              >
                <LogIn className="w-4 h-4" />
                Login
              </button>
            ) : (
              <button
                onClick={() => {
                  void handleSignOut();
                }}
                className="hidden md:inline-flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
              >
                <LogOut className="w-4 h-4" />
                Sign out
              </button>
            )}

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
                <span className="text-xs font-mono text-muted-foreground">System Status:</span>
                <span className="text-xs font-mono font-medium text-accent">
                  {isLoading ? "Processing" : "Optimal"}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-muted-foreground" />
                <span className="text-xs font-mono text-muted-foreground">Active Sources:</span>
                <span className="text-xs font-mono font-medium">
                  {uploadedDocuments.length} indexed PDF(s)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
                <span className="text-xs font-mono text-muted-foreground">{statusMessage}</span>
              </div>
            </div>
          </motion.div>
        )}
      </motion.header>

      <div className="flex-1 flex overflow-hidden relative">
        <AnimatePresence>
          {(showSidebar || isDesktop) && (
            <motion.aside
              initial={{ x: -300, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -300, opacity: 0 }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="w-64 border-r border-border bg-card flex flex-col absolute lg:relative inset-y-0 left-0 z-20"
            >
              <div className="p-4 border-b border-border">
                <button
                  onClick={handleStartNewSession}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-accent text-accent-foreground rounded-lg hover:opacity-90 transition-opacity"
                >
                  <Plus className="w-4 h-4" />
                  <span className="font-medium">New Session</span>
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-2">
                {session?.user?.id ? (
                  conversationHistory.length > 0 ? (
                    <>
                      <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3 px-2">
                        Saved Sessions
                      </div>

                      {conversationHistory.map((conversation) => (
                        <button
                          key={conversation.id}
                          onClick={() => handleOpenConversation(conversation.id)}
                          className={`w-full text-left px-3 py-3 rounded-lg transition-colors ${
                            currentSessionId === conversation.id
                              ? "bg-accent/10 border border-accent/20"
                              : "hover:bg-muted/50 border border-transparent"
                          }`}
                        >
                          <div className="text-sm font-medium mb-1 line-clamp-1">
                            {conversation.title}
                          </div>
                          {/* <div className="text-xs text-muted-foreground line-clamp-2">
                            {conversation.last_message_preview || "Session created"}
                          </div> */}
                          <div className="mt-2 text-[11px] text-muted-foreground/80 font-mono">
                            {new Date(conversation.updated_at).toLocaleDateString("en-US", {
                              month: "short",
                              day: "numeric",
                            })}
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
                  )
                ) : messages.length > 0 ? (
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
                    <button
                      onClick={() => setShowAuthScreen(true)}
                      className="mt-4 inline-flex items-center gap-2 rounded-full border border-accent/20 bg-accent/10 px-4 py-2 text-sm text-accent transition-colors hover:bg-accent/15"
                    >
                      <LogIn className="w-4 h-4" />
                      Login
                    </button>
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
