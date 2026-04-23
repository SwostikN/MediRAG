import { useMemo, useState } from "react";
import { motion } from "motion/react";
import { ArrowLeft, Check, Lock, Mail, MailCheck, ShieldCheck, UserRound, X } from "lucide-react";
import type { KeyboardEvent } from "react";

interface AuthScreenProps {
  onBack: () => void;
  onSignIn: (email: string, password: string) => Promise<void>;
  onSignUp: (
    email: string,
    password: string,
    fullName: string,
  ) => Promise<{ needsConfirmation: boolean }>;
}

type AuthMode = "signin" | "signup";

interface PasswordCheck {
  label: string;
  passed: boolean;
}

function evaluatePassword(pw: string): PasswordCheck[] {
  return [
    { label: "At least 8 characters", passed: pw.length >= 8 },
    { label: "Contains a lowercase letter", passed: /[a-z]/.test(pw) },
    { label: "Contains an uppercase letter", passed: /[A-Z]/.test(pw) },
    { label: "Contains a number", passed: /\d/.test(pw) },
    { label: "Contains a symbol (!@#$…)", passed: /[^A-Za-z0-9]/.test(pw) },
  ];
}

export function AuthScreen({
  onBack,
  onSignIn,
  onSignUp,
}: AuthScreenProps) {
  const [mode, setMode] = useState<AuthMode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [confirmationEmail, setConfirmationEmail] = useState<string | null>(null);

  const passwordChecks = useMemo(() => evaluatePassword(password), [password]);
  const passwordValid = passwordChecks.every((c) => c.passed);

  const switchMode = (nextMode: AuthMode) => {
    setMode(nextMode);
    // Previously, the email / password typed into sign-in persisted when
    // the user flipped to the Create-account tab, which looked like the
    // app was "auto-filling" fields across modes. Reset the form so each
    // tab starts clean.
    setEmail("");
    setPassword("");
    setFullName("");
    setErrorMessage("");
  };

  const humanizeAuthError = (raw: string): string => {
    const lower = raw.toLowerCase();
    if (mode === "signin") {
      // Supabase returns "Invalid login credentials" for BOTH "no
      // account with this email" and "wrong password" — they don't
      // disambiguate for account-enumeration safety. Surface that
      // clearly so first-time users know they need to create an
      // account first instead of assuming their password is wrong.
      if (
        lower.includes("invalid login credentials") ||
        lower.includes("invalid credentials")
      ) {
        return (
          "We couldn't sign you in with this email and password. " +
          "If you don't have an account yet, please create one using " +
          "the Create account tab. If you do, double-check your password."
        );
      }
      if (lower.includes("email not confirmed")) {
        return (
          "Your account exists but the email isn't confirmed yet. " +
          "Check your inbox for the confirmation link."
        );
      }
    }
    if (mode === "signup") {
      if (
        lower.includes("already registered") ||
        lower.includes("user already exists") ||
        lower.includes("email already") ||
        lower.includes("duplicate")
      ) {
        return (
          "An account with this email already exists. Switch to the " +
          "Sign in tab to log in, or use a different email."
        );
      }
    }
    return raw;
  };

  const handleSubmit = async () => {
    setErrorMessage("");

    if (mode === "signup" && !passwordValid) {
      setErrorMessage("Password does not meet all the requirements below.");
      return;
    }

    setIsSubmitting(true);

    try {
      if (mode === "signin") {
        await onSignIn(email, password);
      } else {
        const result = await onSignUp(email, password, fullName);
        if (result.needsConfirmation) {
          setConfirmationEmail(email);
        }
      }
    } catch (error) {
      const raw = error instanceof Error ? error.message : "Authentication failed";
      setErrorMessage(humanizeAuthError(raw));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== "Enter") {
      return;
    }

    event.preventDefault();

    if (isSubmitting) {
      return;
    }

    void handleSubmit();
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
              {confirmationEmail ? (
                <div className="py-6 text-center">
                  <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-full bg-accent/10 text-accent">
                    <MailCheck className="h-7 w-7" />
                  </div>
                  <h2 className="text-2xl font-medium">Check your email</h2>
                  <p className="mt-3 text-sm leading-6 text-muted-foreground">
                    We sent a confirmation link to{" "}
                    <span className="font-medium text-foreground">{confirmationEmail}</span>. Click the link in
                    that email to activate your account, then return here to sign in.
                  </p>
                  <p className="mt-2 text-xs text-muted-foreground/80">
                    Didn’t get it? Check your spam folder, or try signing up again with the same email.
                  </p>
                  <div className="mt-6 flex flex-col gap-2">
                    <button
                      onClick={() => {
                        setConfirmationEmail(null);
                        setMode("signin");
                        setPassword("");
                      }}
                      className="w-full rounded-2xl bg-accent px-4 py-3 text-accent-foreground transition-opacity hover:opacity-90"
                    >
                      Back to sign in
                    </button>
                    <button
                      onClick={onBack}
                      className="w-full rounded-2xl border border-border px-4 py-3 text-sm text-muted-foreground transition-colors hover:text-foreground"
                    >
                      Back to DocuMed AI
                    </button>
                  </div>
                </div>
              ) : (
              <>
              <button
                onClick={onBack}
                className="mb-6 inline-flex items-center gap-2 rounded-full border border-border px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground lg:hidden"
              >
                <ArrowLeft className="h-4 w-4" />
                Back
              </button>

              <div className="mb-6 flex rounded-full bg-muted/60 p-1">
                <button
                  onClick={() => switchMode("signin")}
                  className={`flex-1 rounded-full px-4 py-2 text-sm transition-all ${
                    mode === "signin" ? "bg-background shadow-sm" : "text-muted-foreground"
                  }`}
                >
                  Sign in
                </button>
                <button
                  onClick={() => switchMode("signup")}
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
                      onKeyDown={handleKeyDown}
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
                    onKeyDown={handleKeyDown}
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
                    onKeyDown={handleKeyDown}
                    placeholder={mode === "signup" ? "At least 8 chars, mix of cases, a number, a symbol" : "Your password"}
                    className="w-full rounded-2xl border border-border bg-background px-4 py-3 focus:border-accent focus:outline-none"
                  />
                </label>

                {mode === "signup" && password.length > 0 && (
                  <ul className="space-y-1.5 text-xs">
                    {passwordChecks.map((check) => (
                      <li
                        key={check.label}
                        className={`flex items-center gap-2 ${
                          check.passed ? "text-emerald-600 dark:text-emerald-400" : "text-muted-foreground"
                        }`}
                      >
                        {check.passed ? (
                          <Check className="h-3.5 w-3.5" />
                        ) : (
                          <X className="h-3.5 w-3.5 opacity-60" />
                        )}
                        {check.label}
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              {errorMessage && (
                <div className="mt-5 rounded-2xl border border-red-400/20 bg-red-500/8 px-4 py-3 text-sm text-red-200 backdrop-blur">
                  {errorMessage}
                </div>
              )}

              <button
                onClick={() => {
                  void handleSubmit();
                }}
                disabled={
                  isSubmitting ||
                  !email ||
                  !password ||
                  (mode === "signup" && (!fullName || !passwordValid))
                }
                className="mt-6 w-full rounded-2xl bg-accent px-4 py-3 text-accent-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isSubmitting
                  ? "Please wait..."
                  : mode === "signin"
                    ? "Sign in"
                    : "Create account"}
              </button>
              </>
              )}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
