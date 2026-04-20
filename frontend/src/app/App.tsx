import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import type { Session } from "@supabase/supabase-js";
import {
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
  Trash2,
  X,
} from "lucide-react";
import { ChatMessage, type ChatMessageSource } from "./components/ChatMessage";
import { ChatInput } from "./components/ChatInput";
import { MedicalContextPanel } from "./components/MedicalContextPanel";
import { EmptyState } from "./components/EmptyState";
import { AuthScreen } from "./components/AuthScreen";
import { SettingsSheet } from "./components/SettingsSheet";
import { useSettings } from "../lib/settings";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "./components/ui/alert-dialog";
import { supabase } from "../lib/supabase";

// One parsed lab marker shown in the inline table under a Stage 4
// explainer message. Mirrors backend app.stages.results.LabMarker
// (only the fields the frontend renders).
export interface LabMarker {
  name: string;
  value: number;
  unit: string;
  reference_range: string | null;
  status: "low" | "normal" | "high" | "unknown";
}

interface Message {
  id: number | string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  renderMode?: "plain" | "query";
  redFlag?: {
    ruleId: string;
    category: string;
    urgency: string;
  };
  stage?: string;
  sources?: ChatMessageSource[];
  // Set true when the backend returned coverage:"no_source" (no-coverage
  // refusal). The (preceding-user, this-assistant) pair is dropped from
  // conversation history before the next /query call so an unanswered
  // question doesn't pollute the next retrieval rewrite.
  noCoverage?: boolean;
  // Stage 4 lab-report responses ship a parsed marker list; ChatMessage
  // renders it as a structured table below the prose.
  markers?: LabMarker[];
  // For 'needs_user_intent' upload responses the assistant message
  // shows two buttons that POST /upload/resolve with the user's
  // chosen doc_type. Cleared once a button is clicked.
  resolveActions?: {
    sessionDocId: string;
    filename: string;
  };
}

// Per-session attached document — populated from /upload responses and
// from chat_sessions.attached_documents on session restore.
interface AttachedDoc {
  id: string;
  filename: string;
  doc_type: "lab_report" | "research_paper" | "other";
  page_count?: number | null;
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
  stage?: string | null;
  red_flag?: {
    ruleId: string;
    category: string;
    urgency: string;
  } | null;
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

interface StreamCallbacks {
  onMeta?: (meta: Record<string, unknown>) => void;
  onDelta?: (text: string) => void;
  onSources?: (sources: ChatMessageSource[]) => void;
  onError?: (message: string) => void;
  // Fired when the backend's Week 10 scope-guard fires mid/end-stream.
  // The assistant has already streamed tokens token-by-token that the
  // guard now judges as a diagnosis / prescription; the caller should
  // REPLACE the bubble content with `text` rather than appending to it.
  onOverride?: (override: { text: string; coverage?: string; reason?: string }) => void;
}

// Read a Server-Sent Events stream from /query/stream and dispatch each
// event to the matching callback. Resolves once the stream emits "done"
// or the response body closes. Throws on non-2xx responses or transport
// failure — caller wraps in try/catch.
async function streamQuery(
  url: string,
  body: unknown,
  headers: HeadersInit,
  callbacks: StreamCallbacks,
): Promise<void> {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      ...headers,
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const txt = await response.text();
      if (txt) detail = txt;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  if (!response.body) {
    throw new Error("Streaming response has no body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let done = false;

  while (!done) {
    const { value, done: streamDone } = await reader.read();
    if (streamDone) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE event blocks are separated by a blank line ("\n\n"). The last
    // chunk may be partial — keep it in the buffer until the next read.
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      const trimmed = block.trim();
      if (!trimmed) continue;

      let eventType = "message";
      let dataLine = "";
      for (const line of trimmed.split("\n")) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLine = line.slice(5).trim();
      }
      if (!dataLine) continue;

      let payload: Record<string, unknown>;
      try {
        payload = JSON.parse(dataLine);
      } catch {
        continue;
      }

      if (eventType === "meta") {
        callbacks.onMeta?.(payload);
      } else if (eventType === "delta") {
        callbacks.onDelta?.(typeof payload.text === "string" ? payload.text : "");
      } else if (eventType === "sources") {
        const list = Array.isArray(payload.sources) ? (payload.sources as ChatMessageSource[]) : [];
        callbacks.onSources?.(list);
      } else if (eventType === "override") {
        callbacks.onOverride?.({
          text: typeof payload.text === "string" ? payload.text : "",
          coverage: typeof payload.coverage === "string" ? payload.coverage : undefined,
          reason: typeof payload.reason === "string" ? payload.reason : undefined,
        });
      } else if (eventType === "error") {
        callbacks.onError?.(
          typeof payload.message === "string" ? payload.message : "stream error",
        );
      } else if (eventType === "done") {
        done = true;
      }
    }
  }
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
  // Rich attached-doc list (filename + doc_type), used for the chip
  // strip near the input. uploadedDocuments stays as a flat name list
  // for the existing context-panel readouts.
  const [attachedDocs, setAttachedDocs] = useState<AttachedDoc[]>([]);
  // session_doc_ids currently mid-resolve; used to disable both
  // disambiguation buttons after one is clicked so the request can't
  // fire twice.
  const [resolvingDocs, setResolvingDocs] = useState<Set<string>>(new Set());
  const [statusMessage, setStatusMessage] = useState("No documents indexed yet");
  const [session, setSession] = useState<Session | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [showAuthScreen, setShowAuthScreen] = useState(false);
  const [conversationHistory, setConversationHistory] = useState<ChatSessionRecord[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [historyAvailable, setHistoryAvailable] = useState(true);
  const [deleteTarget, setDeleteTarget] = useState<ChatSessionRecord | null>(null);
  const [deleteInFlight, setDeleteInFlight] = useState(false);
  const [clearAllOpen, setClearAllOpen] = useState(false);
  const [clearAllInFlight, setClearAllInFlight] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [sidebarSearch, setSidebarSearch] = useState("");
  const { t } = useSettings();

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
    setAttachedDocs([]);
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

    const rows = (data ?? []) as ChatMessageRecord[];
    setMessages(toUiMessages(rows));
    setCurrentSessionId(chatSessionId);

    // Backfill title for legacy sessions created with the "New chat"
    // placeholder. Done lazily on open so we don't pay N+1 queries on
    // history load. The first user message is the same seed we would
    // have used on creation, so this matches new-session behaviour.
    const PLACEHOLDER_TITLES = new Set(["New chat", "Untitled session", ""]);
    const existing = conversationHistory.find((s) => s.id === chatSessionId);
    const currentTitle = (existing?.title ?? "").trim();
    if (existing && PLACEHOLDER_TITLES.has(currentTitle)) {
      const firstUser = rows.find((r) => r.role === "user");
      if (firstUser) {
        const newTitle = generateSessionTitle(firstUser.content);
        const { data: renamed } = await supabase
          .from("chat_sessions")
          .update({ title: newTitle })
          .eq("id", chatSessionId)
          .select("id, title, created_at, updated_at, last_message_preview")
          .single();
        if (renamed) {
          upsertConversationHistory(renamed as ChatSessionRecord);
        }
      }
    }

    // Restore the per-session attached documents from the
    // chat_sessions.attached_documents JSONB column. The chunks/markers
    // themselves live in session_chunks / user_lab_markers and are
    // pulled at query time via the session-merge path in
    // _retrieve_ranked, so we don't need to re-index anything here.
    const { data: sessionRow } = await supabase
      .from("chat_sessions")
      .select("attached_documents")
      .eq("id", chatSessionId)
      .maybeSingle();
    const restored = Array.isArray(sessionRow?.attached_documents)
      ? (sessionRow!.attached_documents as AttachedDoc[])
      : [];
    setAttachedDocs(restored);
    setUploadedDocuments(restored.map((d) => d.filename));
    setStatusMessage(
      restored.length
        ? `Session restored with ${restored.length} attached document(s).`
        : "Session restored.",
    );
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

      // Profile sync is telemetry on user_profiles. A failure here
      // (RLS drift, missing column, etc.) must not block history load —
      // chat_sessions is the source of truth for the sidebar and uses
      // the auth-session user_id, not the profile row.
      try {
        await syncUserProfile(activeSession);
      } catch (profileError) {
        console.warn("user_profiles sync failed (continuing)", profileError);
      }
      try {
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
        } catch (profileError) {
          console.warn("user_profiles sync failed (continuing)", profileError);
        }
        try {
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
    message: Pick<Message, "role" | "content" | "renderMode" | "redFlag" | "stage">,
  ) => {
    const now = new Date().toISOString();
    const preview = message.content.slice(0, 140);

    const { error: insertError } = await supabase.from("chat_messages").insert([
      {
        session_id: chatSessionId,
        role: message.role,
        content: message.content,
        render_mode: message.renderMode ?? "plain",
        stage: message.stage ?? null,
        red_flag: message.redFlag
          ? {
              ruleId: message.redFlag.ruleId,
              category: message.redFlag.category,
              urgency: message.redFlag.urgency,
            }
          : null,
      },
    ]);

    if (insertError) {
      console.warn("Supabase message insert error:", insertError.message || insertError);
      return;
    }

    // Rename the session on the first user turn when the stored title is
    // still a placeholder. Backend-created rows (supabase_client.create_chat_session)
    // and older eval sessions all land with title="New chat"; without this
    // patch the sidebar stays a wall of identical labels forever.
    const PLACEHOLDER_TITLES = new Set(["New chat", "Untitled session", ""]);
    const existing = conversationHistory.find((s) => s.id === chatSessionId);
    const shouldRename =
      message.role === "user" &&
      (!existing || PLACEHOLDER_TITLES.has((existing.title ?? "").trim()));

    const updatePayload: Record<string, unknown> = {
      updated_at: now,
      last_message_preview: preview,
    };
    if (shouldRename) {
      updatePayload.title = generateSessionTitle(message.content);
    }

    const { data: updatedSession, error: updateError } = await supabase
      .from("chat_sessions")
      .update(updatePayload)
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

  // Uploads now go through /upload (per-session, never the shared corpus).
  // The endpoint classifies the file (lab_report | research_paper | other)
  // and returns a different shape per branch — this handler maps each
  // shape to one assistant message and updates the attached-docs strip.
  const handleUploadPdf = async (files: File[]) => {
    if (files.length === 0) {
      return;
    }

    if (!session?.user?.id) {
      setStatusMessage("Sign in to upload documents.");
      setShowAuthScreen(true);
      return;
    }

    setIsLoading(true);
    try {
      const chatSessionId = await ensureConversationId(files[0].name);
      if (!chatSessionId) {
        throw new Error(
          "Could not create a conversation to attach this document to. Please sign in again.",
        );
      }

      for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("session_id", chatSessionId);
        formData.append("user_id", session.user.id);

        const uploadResponse = await fetch(`${API_BASE_URL}/upload`, {
          method: "POST",
          headers: buildRequestHeaders(false),
          body: formData,
        });

        const payload = await uploadResponse.json();

        if (!uploadResponse.ok) {
          throw new Error(payload.detail || payload.message || "Failed to upload PDF");
        }

        const status = payload.status as string | undefined;
        const docType = payload.doc_type as
          | "lab_report"
          | "research_paper"
          | "other"
          | undefined;

        // Build the assistant message for this file based on the response
        // shape. Five terminal shapes the backend can return:
        //   ok + lab_report      → markdown explainer + markers table
        //   ok + research_paper  → "indexed N sections" plain msg
        //   needs_user_intent    → "is this lab or paper?" plain msg
        //   duplicate            → "already attached" plain msg
        //   unreadable / off_domain / empty_after_chunking → server message
        let assistantMessage: Message;

        if (status === "ok" && docType === "lab_report") {
          assistantMessage = {
            id: `${Date.now()}-${file.name}`,
            role: "assistant",
            content: payload.answer || "Lab report processed.",
            timestamp: getTimestamp(),
            renderMode: "query",
            stage: "results",
            sources: Array.isArray(payload.sources) ? payload.sources : [],
            markers: Array.isArray(payload.markers) ? payload.markers : [],
          };
        } else if (status === "ok" && docType === "research_paper") {
          assistantMessage = {
            id: `${Date.now()}-${file.name}`,
            role: "assistant",
            content:
              payload.message ||
              `Indexed ${payload.chunks ?? "the"} sections from ${file.name}.`,
            timestamp: getTimestamp(),
            renderMode: "plain",
            stage: "upload",
          };
        } else if (status === "needs_user_intent") {
          assistantMessage = {
            id: `${Date.now()}-${file.name}`,
            role: "assistant",
            content:
              payload.message ||
              "I'm not sure how to handle this document. Tell me whether it's a lab report or a research paper.",
            timestamp: getTimestamp(),
            renderMode: "plain",
            stage: "upload",
            resolveActions: {
              sessionDocId: payload.session_doc_id,
              filename: payload.filename || file.name,
            },
          };
        } else if (status === "duplicate") {
          assistantMessage = {
            id: `${Date.now()}-${file.name}`,
            role: "assistant",
            content:
              payload.message ||
              `${file.name} is already attached to this conversation.`,
            timestamp: getTimestamp(),
            renderMode: "plain",
            stage: "upload",
          };
        } else {
          // unreadable | off_domain | empty_after_chunking | other errors
          assistantMessage = {
            id: `${Date.now()}-${file.name}`,
            role: "assistant",
            content:
              payload.message ||
              `I couldn't process ${file.name}. Please try a different file.`,
            timestamp: getTimestamp(),
            renderMode: "plain",
            stage: "upload",
          };
        }

        setMessages((prev) => [...prev, assistantMessage]);
        if (chatSessionId) {
          void persistConversationMessage(chatSessionId, assistantMessage);
        }

        // Update the attached-docs strip for any response that produced
        // a session_documents row (everything except read-failures).
        if (
          payload.session_doc_id &&
          (status === "ok" ||
            status === "duplicate" ||
            status === "needs_user_intent")
        ) {
          const entry: AttachedDoc = {
            id: payload.session_doc_id,
            filename: payload.filename || file.name,
            doc_type: (docType as AttachedDoc["doc_type"]) || "other",
            page_count: payload.page_count ?? null,
          };
          setAttachedDocs((prev) => {
            if (prev.some((d) => d.id === entry.id)) return prev;
            return [...prev, entry];
          });
          setUploadedDocuments((prev) =>
            Array.from(new Set([...prev, entry.filename])),
          );
        }
      }

      setStatusMessage(`Processed ${files.length} file(s).`);
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

  // Fires when the user clicks "Treat as lab report" / "Treat as research
  // paper" under a needs_user_intent upload message. Calls /upload/resolve,
  // then rewrites the same assistant message in-place with the resolved
  // result (markers + explainer, or "indexed N sections") and strips the
  // buttons. Also flips the doc-type chip colour in attachedDocs from grey.
  const handleResolveUpload = async (
    sessionDocId: string,
    docType: "lab_report" | "research_paper",
    messageId: number | string,
  ) => {
    if (!session?.user?.id || !currentSessionId) {
      setStatusMessage("Sign in to continue.");
      return;
    }
    if (resolvingDocs.has(sessionDocId)) return;

    setResolvingDocs((prev) => {
      const next = new Set(prev);
      next.add(sessionDocId);
      return next;
    });

    try {
      const response = await fetch(`${API_BASE_URL}/upload/resolve`, {
        method: "POST",
        headers: buildRequestHeaders(true),
        body: JSON.stringify({
          session_doc_id: sessionDocId,
          session_id: currentSessionId,
          user_id: session.user.id,
          doc_type: docType,
        }),
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail || payload.message || "Resolve failed");
      }

      const status = payload.status as string | undefined;
      const returnedType = payload.doc_type as
        | "lab_report"
        | "research_paper"
        | undefined;

      setMessages((prev) =>
        prev.map((m) => {
          if (m.id !== messageId) return m;
          if (status === "ok" && returnedType === "lab_report") {
            return {
              ...m,
              content: payload.answer || "Lab report processed.",
              renderMode: "query",
              stage: "results",
              sources: Array.isArray(payload.sources) ? payload.sources : [],
              markers: Array.isArray(payload.markers) ? payload.markers : [],
              resolveActions: undefined,
            };
          }
          if (status === "ok" && returnedType === "research_paper") {
            return {
              ...m,
              content:
                payload.message ||
                `Indexed ${payload.chunks ?? "the"} sections.`,
              renderMode: "plain",
              stage: "upload",
              resolveActions: undefined,
            };
          }
          return {
            ...m,
            content:
              payload.message ||
              "Could not process this document as the selected type.",
            renderMode: "plain",
            stage: "upload",
            resolveActions: status === "off_domain" ? undefined : m.resolveActions,
          };
        }),
      );

      if (status === "ok" && returnedType) {
        setAttachedDocs((prev) =>
          prev.map((d) =>
            d.id === sessionDocId ? { ...d, doc_type: returnedType } : d,
          ),
        );
        setStatusMessage(
          returnedType === "lab_report"
            ? "Lab report processed."
            : `Indexed ${payload.chunks ?? ""} sections.`.trim(),
        );
      }
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Could not resolve this document.";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === messageId
            ? { ...m, content: `${m.content}\n\nError: ${message}` }
            : m,
        ),
      );
      setStatusMessage("Resolve failed");
    } finally {
      setResolvingDocs((prev) => {
        const next = new Set(prev);
        next.delete(sessionDocId);
        return next;
      });
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
      // Send the last few turns as conversation context so follow-ups
      // ("what are the symptoms?") inherit the topic of earlier turns
      // ("hypertension"). Drop *pairs* (user-question + its assistant
      // response) that we don't want polluting the next retrieval rewrite:
      //   - red-flag emergency templates
      //   - error placeholders
      //   - no-coverage refusals (otherwise "who is a gynac?" gets
      //     concatenated into the next retrieval query and drags the
      //     rerank score below the gate threshold for a follow-up that
      //     would otherwise have answered correctly)
      // Refusal text is also matched on content for the case where messages
      // were rehydrated from Supabase (chat_messages doesn't persist the
      // coverage marker, so the flag is missing on resumed sessions).
      const REFUSAL_PREFIX = "I don't have a source for that in my current library";
      const skipAssistant = (m: Message) =>
        Boolean(m.redFlag) ||
        m.noCoverage === true ||
        (m.role === "assistant" && m.content.startsWith("Error:")) ||
        (m.role === "assistant" && m.content.startsWith(REFUSAL_PREFIX));
      const skipIndices = new Set<number>();
      messages.forEach((m, i) => {
        if (m.role === "assistant" && skipAssistant(m)) {
          skipIndices.add(i);
          if (i > 0 && messages[i - 1].role === "user") {
            skipIndices.add(i - 1);
          }
        }
      });
      const historyTurns = messages
        .filter((_, i) => !skipIndices.has(i))
        .slice(-6)
        .map((m) => ({ role: m.role, content: m.content }));

      const queryBody: {
        question: string;
        session_id?: string;
        history?: { role: "user" | "assistant"; content: string }[];
      } = {
        question: trimmedContent,
      };
      if (chatSessionId) {
        queryBody.session_id = chatSessionId;
      }
      if (historyTurns.length > 0) {
        queryBody.history = historyTurns;
      }

      // Streaming path. We DON'T insert the assistant bubble immediately —
      // if we did, it would render as an empty bubble alongside the
      // "Contacting DocuMed AI backend..." dots loader (line ~1160), which
      // looks like two separate things happening at once. Instead, the
      // dots loader stays visible during the wait, and we add the bubble
      // only when the first delta arrives. The dots disappear and the
      // answer appears in the same render pass, in the same place.
      //
      // Meta info (stage, red_flag, coverage) may arrive BEFORE the first
      // delta — we buffer it in `pendingMeta` and apply it when we
      // finally create the bubble.
      const assistantMessageId = Date.now() + 2;
      let bubbleAdded = false;
      let pendingMeta: Partial<Message> = {};
      let pendingSources: ChatMessageSource[] | undefined;
      let finalMessage: Message = {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        timestamp: getTimestamp(),
        renderMode: "query",
      };

      const ensureBubble = (initialContent: string) => {
        if (bubbleAdded) return;
        bubbleAdded = true;
        finalMessage = {
          ...finalMessage,
          ...pendingMeta,
          content: initialContent,
          sources: pendingSources,
        };
        const snapshot = finalMessage;
        setMessages((prev) => [...prev, snapshot]);
      };

      await streamQuery(
        `${API_BASE_URL}/query/stream`,
        queryBody,
        buildRequestHeaders(true),
        {
          onMeta: (meta) => {
            const stage = typeof meta.stage === "string" ? meta.stage : undefined;
            const noCoverage = meta.coverage === "no_source" ? true : undefined;
            const rf = meta.red_flag as
              | { rule_id?: string; category?: string; urgency?: string }
              | undefined;
            const redFlag = rf
              ? {
                  ruleId: rf.rule_id ?? "",
                  category: rf.category ?? "",
                  urgency: rf.urgency ?? "",
                }
              : undefined;
            pendingMeta = { stage, noCoverage, redFlag };
            if (bubbleAdded) {
              finalMessage = { ...finalMessage, ...pendingMeta };
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMessageId ? { ...m, ...pendingMeta } : m,
                ),
              );
            }
          },
          onDelta: (text) => {
            if (!text) return;
            if (!bubbleAdded) {
              ensureBubble(text);
              return;
            }
            finalMessage = { ...finalMessage, content: finalMessage.content + text };
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId ? { ...m, content: m.content + text } : m,
              ),
            );
          },
          onSources: (sources) => {
            if (!bubbleAdded) {
              pendingSources = sources;
              return;
            }
            finalMessage = { ...finalMessage, sources };
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantMessageId ? { ...m, sources } : m)),
            );
          },
          // Scope-guard override: backend decided the streamed answer is
          // out of scope (diagnosis / prescription). Replace whatever we
          // already streamed into the bubble with the refusal text and
          // strip sources so the bubble renders as a refusal. `noCoverage`
          // also makes `skipAssistant` drop this pair from the next
          // retrieval rewrite (same treatment as other refusals).
          onOverride: ({ text }) => {
            if (!text) return;
            if (!bubbleAdded) {
              pendingSources = undefined;
              ensureBubble(text);
              finalMessage = { ...finalMessage, noCoverage: true, sources: undefined };
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMessageId
                    ? { ...m, content: text, noCoverage: true, sources: undefined }
                    : m,
                ),
              );
              return;
            }
            finalMessage = {
              ...finalMessage,
              content: text,
              noCoverage: true,
              sources: undefined,
            };
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId
                  ? { ...m, content: text, noCoverage: true, sources: undefined }
                  : m,
              ),
            );
          },
          onError: (msg) => {
            const note = `\n\n_${msg}_`;
            if (!bubbleAdded) {
              ensureBubble(note.trimStart());
              return;
            }
            finalMessage = { ...finalMessage, content: finalMessage.content + note };
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId ? { ...m, content: m.content + note } : m,
              ),
            );
          },
        },
      );

      // Stream closed without ever sending a delta or error. Surface
      // something rather than leaving the dots loader stranded.
      if (!bubbleAdded) {
        ensureBubble("(no response)");
      }

      setStatusMessage("Backend response received");

      if (chatSessionId) {
        void persistConversationMessage(chatSessionId, finalMessage);
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

  // Trash-icon click: stage the row for confirmation. The actual delete
  // runs from the AlertDialog's Continue button (confirmDeleteConversation).
  const requestDeleteConversation = (conversation: ChatSessionRecord) => {
    if (!session?.user?.id) return;
    setDeleteTarget(conversation);
  };

  // Hard-delete a chat session and everything scoped to it. The DB does
  // the heavy lifting via on-delete-cascade:
  //   chat_sessions → chat_messages
  //   chat_sessions → session_documents → session_chunks, user_lab_markers
  // query_log.session_id has no FK (text column), so we sweep it by hand
  // for privacy parity. RLS already scopes every table to the signed-in
  // user, so the delete can only touch the caller's own rows.
  // Nuke every chat_sessions row owned by the signed-in user. Cascades
  // handle chat_messages, session_documents → session_chunks +
  // user_lab_markers. query_log is swept by user_id (no FK on the
  // session_id text column). Testing-phase shortcut — guarded by a
  // typed-confirm dialog so accidental clicks don't wipe everything.
  const confirmClearAllConversations = async () => {
    if (!session?.user?.id) return;
    setClearAllInFlight(true);

    const userId = session.user.id;
    const { error } = await supabase
      .from("chat_sessions")
      .delete()
      .eq("user_id", userId);

    if (error) {
      console.warn("Supabase clear-all error:", error.message || error);
      setStatusMessage(`Could not clear history: ${error.message ?? "unknown error"}`);
      setClearAllInFlight(false);
      setClearAllOpen(false);
      return;
    }

    void supabase.from("query_log").delete().eq("user_id", userId);

    setConversationHistory([]);
    resetLocalConversation("All chats cleared.");
    setClearAllInFlight(false);
    setClearAllOpen(false);
  };

  const confirmDeleteConversation = async () => {
    if (!deleteTarget || !session?.user?.id) return;
    const chatSessionId = deleteTarget.id;
    setDeleteInFlight(true);

    const { error } = await supabase
      .from("chat_sessions")
      .delete()
      .eq("id", chatSessionId);

    if (error) {
      console.warn("Supabase session delete error:", error.message || error);
      setStatusMessage(`Could not delete chat: ${error.message ?? "unknown error"}`);
      setDeleteInFlight(false);
      setDeleteTarget(null);
      return;
    }

    // Best-effort telemetry sweep. Failure here must not block the UI
    // update — the user-visible data is already gone.
    void supabase.from("query_log").delete().eq("session_id", chatSessionId);

    setConversationHistory((prev) => prev.filter((s) => s.id !== chatSessionId));
    if (currentSessionId === chatSessionId) {
      resetLocalConversation("Chat deleted.");
    } else {
      setStatusMessage("Chat deleted.");
    }
    setDeleteInFlight(false);
    setDeleteTarget(null);
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

  const handleSignUp = async (
    email: string,
    password: string,
    fullName: string,
  ): Promise<{ needsConfirmation: boolean }> => {
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
      return { needsConfirmation: false };
    }

    setStatusMessage("Account created. Check your email to confirm, then sign in.");
    return { needsConfirmation: true };
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
            <button
              onClick={() => setShowSidebar(!showSidebar)}
              className="lg:hidden p-2 hover:bg-muted rounded-lg transition-colors"
            >
              <Menu className="w-5 h-5" />
            </button>

            <button
              onClick={handleStartNewSession}
              className="flex items-center gap-20 rounded-lg transition-colors hover:bg-muted/40"
              aria-label="Go to home screen"
            >
              <img
                src="../Documed_Logo.png"
                alt="DocuMed AI Logo"
                className="w-40 h-16 rounded-lg object-contain"
                style={{ transform: "scale(1.8)", transformOrigin: "left center" }}
              />
              <div className="text-left">
                <h1 className="text-lg font-medium">DocuMed AI</h1>
                <p className="text-xs text-muted-foreground font-mono">{t("app_subtitle")}</p>
                <p className="text-xs text-muted-foreground/80">{authLabel}</p>
              </div>
            </button>
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
                {t("tab_classic")}
              </button>
              <button
                onClick={() => setVariant("diagnostic")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  variant === "diagnostic"
                    ? "bg-background shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {t("tab_diagnostic")}
              </button>
              <button
                onClick={() => setVariant("research")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  variant === "research"
                    ? "bg-background shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {t("tab_research")}
              </button>
            </div>

            {!session ? (
              <button
                onClick={() => setShowAuthScreen(true)}
                className="hidden md:inline-flex items-center gap-2 rounded-lg border border-accent/20 bg-accent/10 px-3 py-2 text-sm text-accent transition-colors hover:bg-accent/15"
              >
                <LogIn className="w-4 h-4" />
                {t("login")}
              </button>
            ) : (
              <button
                onClick={() => {
                  void handleSignOut();
                }}
                className="hidden md:inline-flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
              >
                <LogOut className="w-4 h-4" />
                {t("sign_out")}
              </button>
            )}

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

            <button
              onClick={() => setShowSettings(true)}
              aria-label={t("settings")}
              title={t("settings")}
              className="p-2 hover:bg-muted rounded-lg transition-colors"
            >
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
              <div className="p-4 border-b border-border space-y-3">
                <button
                  onClick={handleStartNewSession}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-accent text-accent-foreground rounded-lg hover:opacity-90 transition-opacity"
                >
                  <Plus className="w-4 h-4" />
                  <span className="font-medium">{t("new_session")}</span>
                </button>

                {session?.user?.id && conversationHistory.length > 0 && (
                  <div className="relative">
                    <Search className="w-4 h-4 absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none" />
                    <input
                      type="text"
                      value={sidebarSearch}
                      onChange={(e) => setSidebarSearch(e.target.value)}
                      placeholder={t("search_placeholder")}
                      className="w-full pl-8 pr-8 py-2 bg-muted/40 border border-border rounded-lg text-sm placeholder:text-muted-foreground/70 focus:outline-none focus:border-accent/50 focus:bg-background transition-colors"
                    />
                    {sidebarSearch && (
                      <button
                        onClick={() => setSidebarSearch("")}
                        aria-label="Clear search"
                        className="absolute right-1.5 top-1/2 -translate-y-1/2 p-1 rounded hover:bg-muted transition-colors"
                      >
                        <X className="w-3.5 h-3.5 text-muted-foreground" />
                      </button>
                    )}
                  </div>
                )}
              </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-2">
                {session?.user?.id ? (
                  conversationHistory.length > 0 ? (
                    (() => {
                      const q = sidebarSearch.trim().toLowerCase();
                      const filtered = q
                        ? conversationHistory.filter((c) => {
                            const hay = `${c.title ?? ""} ${c.last_message_preview ?? ""}`.toLowerCase();
                            return hay.includes(q);
                          })
                        : conversationHistory;
                      return (
                        <>
                          <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3 px-2">
                            {t("saved_sessions")}
                          </div>

                          {filtered.length === 0 ? (
                            <div className="px-2 py-6 text-center text-xs text-muted-foreground">
                              {t("no_chats_match")} “{sidebarSearch}”.
                            </div>
                          ) : (
                            filtered.map((conversation) => (
                        <div
                          key={conversation.id}
                          className={`group relative w-full rounded-lg transition-colors ${
                            currentSessionId === conversation.id
                              ? "bg-accent/10 border border-accent/20"
                              : "hover:bg-muted/50 border border-transparent"
                          }`}
                        >
                          <button
                            onClick={() => handleOpenConversation(conversation.id)}
                            className="w-full text-left px-3 py-3 pr-10"
                          >
                            <div className="text-sm font-medium mb-1 line-clamp-1">
                              {conversation.title}
                            </div>
                            <div className="mt-2 text-[11px] text-muted-foreground/80 font-mono">
                              {new Date(conversation.updated_at).toLocaleDateString("en-US", {
                                month: "short",
                                day: "numeric",
                              })}
                            </div>
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              requestDeleteConversation(conversation);
                            }}
                            aria-label={`Delete chat: ${conversation.title}`}
                            title="Delete chat"
                            className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-md text-muted-foreground/70 opacity-0 group-hover:opacity-100 focus:opacity-100 hover:bg-destructive/10 hover:text-destructive transition-opacity transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                            ))
                          )}
                        </>
                      );
                    })()
                  ) : (
                    <div className="flex flex-col items-center justify-center h-full text-center px-4">
                      <FileText className="w-12 h-12 text-muted-foreground/30 mb-4" />
                      <p className="text-sm text-muted-foreground">{t("no_sessions")}</p>
                      <p className="text-xs text-muted-foreground/70 mt-1">
                        {t("upload_pdf_start")}
                      </p>
                    </div>
                  )
                ) : messages.length > 0 ? (
                  (() => {
                    const firstUserMessage = messages.find((m) => m.role === "user");
                    if (!firstUserMessage) return null;
                    return (
                      <>
                        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3 px-2">
                          {t("current_session")}
                        </div>
                        <button className="w-full text-left px-3 py-3 rounded-lg bg-accent/10 border border-accent/20 transition-colors">
                          <div className="text-sm font-medium mb-1 line-clamp-2">
                            {firstUserMessage.content}
                          </div>
                          <div className="mt-2 text-[11px] text-muted-foreground/80 font-mono">
                            {firstUserMessage.timestamp}
                          </div>
                        </button>
                        <p className="mt-3 px-2 text-[11px] leading-5 text-muted-foreground/80">
                          {t("sign_in_to_keep")}
                        </p>
                      </>
                    );
                  })()
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-center px-4">
                    <FileText className="w-12 h-12 text-muted-foreground/30 mb-4" />
                    <p className="text-sm text-muted-foreground">{t("no_sessions")}</p>
                    <p className="text-xs text-muted-foreground/70 mt-1">
                      {t("upload_pdf_start")}
                    </p>
                    <button
                      onClick={() => setShowAuthScreen(true)}
                      className="mt-4 inline-flex items-center gap-2 rounded-full border border-accent/20 bg-accent/10 px-4 py-2 text-sm text-accent transition-colors hover:bg-accent/15"
                    >
                      <LogIn className="w-4 h-4" />
                      {t("login")}
                    </button>
                  </div>
                )}
              </div>

              {session?.user?.id && conversationHistory.length > 0 && (
                <div className="border-t border-border p-3">
                  <button
                    onClick={() => setClearAllOpen(true)}
                    className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg border border-destructive/30 bg-destructive/5 text-destructive text-sm font-medium transition-colors hover:bg-destructive/10 hover:border-destructive/50"
                  >
                    <Trash2 className="w-4 h-4" />
                    Clear all chats
                  </button>
                  <p className="mt-2 text-[10px] leading-4 text-muted-foreground/70 text-center uppercase tracking-wider">
                    {/* Testing · deletes all {conversationHistory.length} sessions */}
                  </p>
                </div>
              )}
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
                  onUploadClick={() => {
                    if (!session?.user?.id) {
                      setStatusMessage("Sign in to upload documents.");
                      setShowAuthScreen(true);
                      return false;
                    }
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
                        : t("input_placeholder_default")
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
                      <ChatMessage
                        key={message.id}
                        {...message}
                        onResolveUpload={(sid, dt) =>
                          void handleResolveUpload(sid, dt, message.id)
                        }
                        resolvePending={
                          message.resolveActions
                            ? resolvingDocs.has(message.resolveActions.sessionDocId)
                            : false
                        }
                      />
                    ))}
                  </AnimatePresence>

                  {isLoading && (messages.length === 0 || messages[messages.length - 1].role === "user") && (
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
                              Contacting DocuMed AI Doctor...
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
                  {attachedDocs.length > 0 && (
                    <div className="mb-3 flex flex-wrap gap-2">
                      {attachedDocs.map((doc) => {
                        const typeLabel =
                          doc.doc_type === "lab_report"
                            ? "Lab"
                            : doc.doc_type === "research_paper"
                              ? "Paper"
                              : "Doc";
                        const typeChipClass =
                          doc.doc_type === "lab_report"
                            ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-200"
                            : doc.doc_type === "research_paper"
                              ? "bg-sky-100 text-sky-700 dark:bg-sky-950/60 dark:text-sky-200"
                              : "bg-muted text-muted-foreground";
                        return (
                          <div
                            key={doc.id}
                            className="flex items-center gap-2 rounded-full border border-border/70 bg-card/90 px-3 py-1 text-xs shadow-sm"
                            title={`${doc.filename}${doc.page_count ? ` · ${doc.page_count} pages` : ""}`}
                          >
                            <span
                              className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${typeChipClass}`}
                            >
                              {typeLabel}
                            </span>
                            <span className="max-w-[180px] truncate font-medium text-foreground">
                              {doc.filename}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  <ChatInput
                    onSend={(message) => {
                      void handleSendMessage(message);
                    }}
                    onUpload={(files) => {
                      void handleUploadPdf(files);
                    }}
                    onUploadClick={() => {
                      if (!session?.user?.id) {
                        setStatusMessage("Sign in to upload documents.");
                        setShowAuthScreen(true);
                        return false;
                      }
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

      <AlertDialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open && !deleteInFlight) {
            setDeleteTarget(null);
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this chat?</AlertDialogTitle>
            <AlertDialogDescription>
              <span className="block">
                "{(deleteTarget?.title ?? "").trim() || "Untitled chat"}" and all its
                messages, uploaded documents, and parsed lab markers will be
                permanently removed. This cannot be undone.
              </span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleteInFlight}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              disabled={deleteInFlight}
              onClick={(e) => {
                e.preventDefault();
                void confirmDeleteConversation();
              }}
              className="bg-destructive text-white hover:bg-destructive/90 focus-visible:ring-destructive"
            >
              {deleteInFlight ? "Deleting..." : "Delete chat"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog
        open={clearAllOpen}
        onOpenChange={(open) => {
          if (!open && !clearAllInFlight) {
            setClearAllOpen(false);
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Clear all chat history?</AlertDialogTitle>
            <AlertDialogDescription>
              <span className="block">
                This will permanently delete all {conversationHistory.length} saved
                sessions, every message in them, and any uploaded documents or lab
                markers attached to those sessions. This cannot be undone.
              </span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={clearAllInFlight}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              disabled={clearAllInFlight}
              onClick={(e) => {
                e.preventDefault();
                void confirmClearAllConversations();
              }}
              className="bg-destructive text-white hover:bg-destructive/90 focus-visible:ring-destructive"
            >
              {clearAllInFlight ? "Clearing..." : "Clear everything"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <SettingsSheet
        open={showSettings}
        onOpenChange={setShowSettings}
        signedIn={!!session?.user?.id}
        onClearAllChats={() => setClearAllOpen(true)}
      />
    </div>
  );
}
