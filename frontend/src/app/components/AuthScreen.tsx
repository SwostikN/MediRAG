import { useState } from "react";
import { motion } from "motion/react";
import { ArrowLeft, Lock, Mail, ShieldCheck, UserRound } from "lucide-react";

interface AuthScreenProps {
  onBack: () => void;
  onSignIn: (email: string, password: string) => Promise<void>;
  onSignUp: (email: string, password: string, fullName: string) => Promise<void>;
  statusMessage?: string;
}

type AuthMode = "signin" | "signup";

export function AuthScreen({
  onBack,
  onSignIn,
  onSignUp,
  statusMessage,
}: AuthScreenProps) {
  const [mode, setMode] = useState<AuthMode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const handleSubmit = async () => {
    setErrorMessage("");
    setIsSubmitting(true);

    try {
      if (mode === "signin") {
        await onSignIn(email, password);
      } else {
        await onSignUp(email, password, fullName);
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Authentication failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground overflow-hidden">
      <div
        className="fixed inset-0 opacity-[0.035] pointer-events-none"
        style={{
          backgroundImage:
            "radial-gradient(circle at 20% 20%, rgba(0, 191, 165, 0.16), transparent 28%), radial-gradient(circle at 80% 0%, rgba(59, 130, 246, 0.12), transparent 22%), radial-gradient(circle at 50% 100%, rgba(0, 191, 165, 0.1), transparent 28%)",
        }}
      />

      <div className="relative z-10 flex min-h-screen items-center justify-center px-6 py-10">
        <div className="w-full max-w-6xl grid gap-8 lg:grid-cols-[1.1fr_0.9fr] items-center">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45 }}
            className="hidden lg:block"
          >
            <button
              onClick={onBack}
              className="mb-8 inline-flex items-center gap-2 rounded-full border border-border bg-card/70 px-4 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to DocuMed AI
            </button>

            <div className="max-w-xl">
              <div className="mb-6 inline-flex items-center gap-3 rounded-full border border-accent/20 bg-accent/10 px-4 py-2 text-sm text-accent">
                <ShieldCheck className="h-4 w-4" />
                Secure medical chat history with DocuMed AI
              </div>
              <h1 className="mb-5 text-5xl font-medium leading-tight">
                Sign in to keep each conversation tied to the right clinician.
              </h1>
              <p className="mb-10 max-w-lg text-lg leading-8 text-muted-foreground">
                Your sessions, prompts, and answers stay organized per user so the app feels
                more like a persistent workspace and less like a temporary demo.
              </p>

              <div className="grid gap-4 sm:grid-cols-3">
                {[
                  ["Per-user sessions", "Each login gets its own chat timeline."],
                  ["Saved responses", "Queries and answers reload when users return."],
                  ["Instant Care", "Get trusted medical answers, symptom guidance, and clear next steps in seconds."],
                ].map(([title, description]) => (
                  <div
                    key={title}
                    className="rounded-2xl border border-border bg-card/80 p-5 shadow-[0_18px_50px_rgba(15,23,42,0.08)]"
                  >
                    <div className="mb-2 text-sm font-medium">{title}</div>
                    <p className="text-sm leading-6 text-muted-foreground">{description}</p>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.08 }}
            className="mx-auto w-full max-w-lg"
          >
            <div className="rounded-[2rem] border border-border bg-card/95 p-8 shadow-[0_30px_80px_rgba(15,23,42,0.14)] backdrop-blur">
              <button
                onClick={onBack}
                className="mb-6 inline-flex items-center gap-2 rounded-full border border-border px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground lg:hidden"
              >
                <ArrowLeft className="h-4 w-4" />
                Back
              </button>

              <div className="mb-6 flex rounded-full bg-muted/60 p-1">
                <button
                  onClick={() => setMode("signin")}
                  className={`flex-1 rounded-full px-4 py-2 text-sm transition-all ${
                    mode === "signin" ? "bg-background shadow-sm" : "text-muted-foreground"
                  }`}
                >
                  Sign in
                </button>
                <button
                  onClick={() => setMode("signup")}
                  className={`flex-1 rounded-full px-4 py-2 text-sm transition-all ${
                    mode === "signup" ? "bg-background shadow-sm" : "text-muted-foreground"
                  }`}
                >
                  Create account
                </button>
              </div>

              <div className="mb-8">
                <h2 className="text-2xl font-medium">
                  {mode === "signin" ? "Welcome back" : "Create your workspace"}
                </h2>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                  {mode === "signin"
                    ? "Use your Supabase account to restore chat history."
                    : "New users get a private profile and persistent conversation history."}
                </p>
              </div>

              <div className="space-y-4">
                {mode === "signup" && (
                  <label className="block">
                    <span className="mb-2 flex items-center gap-2 text-sm text-muted-foreground">
                      <UserRound className="h-4 w-4" />
                      Full name
                    </span>
                    <input
                      type="text"
                      value={fullName}
                      onChange={(event) => setFullName(event.target.value)}
                      placeholder="Dr. Jane Doe"
                      className="w-full rounded-2xl border border-border bg-background px-4 py-3 focus:border-accent focus:outline-none"
                    />
                  </label>
                )}

                <label className="block">
                  <span className="mb-2 flex items-center gap-2 text-sm text-muted-foreground">
                    <Mail className="h-4 w-4" />
                    Email
                  </span>
                  <input
                    type="email"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    placeholder="doctor@hospital.org"
                    className="w-full rounded-2xl border border-border bg-background px-4 py-3 focus:border-accent focus:outline-none"
                  />
                </label>

                <label className="block">
                  <span className="mb-2 flex items-center gap-2 text-sm text-muted-foreground">
                    <Lock className="h-4 w-4" />
                    Password
                  </span>
                  <input
                    type="password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    placeholder="Minimum 6 characters"
                    className="w-full rounded-2xl border border-border bg-background px-4 py-3 focus:border-accent focus:outline-none"
                  />
                </label>
              </div>

              {(errorMessage || statusMessage) && (
                <div className="mt-5 rounded-2xl border border-border bg-muted/40 px-4 py-3 text-sm text-muted-foreground">
                  {errorMessage || statusMessage}
                </div>
              )}

              <button
                onClick={() => {
                  void handleSubmit();
                }}
                disabled={isSubmitting || !email || !password || (mode === "signup" && !fullName)}
                className="mt-6 w-full rounded-2xl bg-accent px-4 py-3 text-accent-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isSubmitting
                  ? "Please wait..."
                  : mode === "signin"
                    ? "Sign in"
                    : "Create account"}
              </button>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
