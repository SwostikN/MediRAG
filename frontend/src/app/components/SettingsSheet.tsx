import { useState } from "react";
import { Globe, Type, Shield, Trash2, UserX, FileText } from "lucide-react";
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "./ui/sheet";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { useSettings, type Language, type TextSize } from "../../lib/settings";

interface SettingsSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  signedIn: boolean;
  onClearAllChats: () => void;
  onDeleteAccount?: () => void;
  privacyUrl?: string;
}

const LANGUAGES: { value: Language; label: string }[] = [
  { value: "en", label: "English" },
  { value: "ne", label: "नेपाली" },
];

const TEXT_SIZES: { value: TextSize; key: "text_size_small" | "text_size_medium" | "text_size_large" }[] = [
  { value: "small", key: "text_size_small" },
  { value: "medium", key: "text_size_medium" },
  { value: "large", key: "text_size_large" },
];

export function SettingsSheet({
  open,
  onOpenChange,
  signedIn,
  onClearAllChats,
  onDeleteAccount,
  privacyUrl,
}: SettingsSheetProps) {
  const { language, setLanguage, textSize, setTextSize, t } = useSettings();
  const [privacyOpen, setPrivacyOpen] = useState(false);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-md flex flex-col">
        <SheetHeader className="text-left">
          <SheetTitle>{t("settings")}</SheetTitle>
        </SheetHeader>

        <div className="flex-1 overflow-y-auto px-4 pb-6 space-y-8">
          {/* Language */}
          <section>
            <div className="flex items-center gap-2 mb-3">
              <Globe className="w-4 h-4 text-accent" />
              <h3 className="text-sm font-medium">{t("language")}</h3>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {LANGUAGES.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setLanguage(opt.value)}
                  className={`px-3 py-2.5 rounded-lg border text-sm transition-colors ${
                    language === opt.value
                      ? "border-accent bg-accent/10 text-accent"
                      : "border-border hover:bg-muted/50"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </section>

          {/* Text size */}
          <section>
            <div className="flex items-center gap-2 mb-3">
              <Type className="w-4 h-4 text-accent" />
              <h3 className="text-sm font-medium">{t("text_size")}</h3>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {TEXT_SIZES.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setTextSize(opt.value)}
                  className={`px-3 py-2.5 rounded-lg border text-sm transition-colors ${
                    textSize === opt.value
                      ? "border-accent bg-accent/10 text-accent"
                      : "border-border hover:bg-muted/50"
                  }`}
                >
                  {t(opt.key)}
                </button>
              ))}
            </div>
          </section>

          {/* Data & Privacy */}
          <section>
            <div className="flex items-center gap-2 mb-3">
              <Shield className="w-4 h-4 text-accent" />
              <h3 className="text-sm font-medium">{t("data_privacy")}</h3>
            </div>

            <div className="space-y-2">
              <button
                onClick={() => {
                  onOpenChange(false);
                  onClearAllChats();
                }}
                disabled={!signedIn}
                className="w-full flex items-start gap-3 px-3 py-3 rounded-lg border border-border text-left hover:bg-muted/50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Trash2 className="w-4 h-4 mt-0.5 text-muted-foreground" />
                <div className="flex-1">
                  <div className="text-sm font-medium">{t("clear_all_chats")}</div>
                  <div className="text-xs text-muted-foreground mt-0.5">
                    {t("clear_all_chats_note")}
                  </div>
                </div>
              </button>

              <button
                onClick={onDeleteAccount}
                disabled={!signedIn || !onDeleteAccount}
                className="w-full flex items-start gap-3 px-3 py-3 rounded-lg border border-border text-left hover:bg-destructive/5 hover:border-destructive/30 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <UserX className="w-4 h-4 mt-0.5 text-muted-foreground" />
                <div className="flex-1">
                  <div className="text-sm font-medium">{t("delete_account")}</div>
                  <div className="text-xs text-muted-foreground mt-0.5">
                    {signedIn ? t("delete_account_note") : t("delete_account_note")}
                  </div>
                </div>
              </button>

              {privacyUrl ? (
                // External URL (e.g. a hosted privacy policy). When supplied,
                // open in a new tab. When not supplied (default), show the
                // in-app disclaimer dialog instead — this prevents the
                // previous "/privacy" href from routing the SPA to the
                // catch-all handler and bouncing the user to EmptyState.
                <a
                  href={privacyUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-full flex items-start gap-3 px-3 py-3 rounded-lg border border-border hover:bg-muted/50 transition-colors"
                >
                  <FileText className="w-4 h-4 mt-0.5 text-muted-foreground" />
                  <div className="flex-1">
                    <div className="text-sm font-medium">{t("privacy_policy")}</div>
                    <div className="text-xs text-muted-foreground mt-0.5">
                      {t("privacy_policy_note")}
                    </div>
                  </div>
                </a>
              ) : (
                <button
                  type="button"
                  onClick={() => setPrivacyOpen(true)}
                  className="w-full flex items-start gap-3 px-3 py-3 rounded-lg border border-border text-left hover:bg-muted/50 transition-colors"
                >
                  <FileText className="w-4 h-4 mt-0.5 text-muted-foreground" />
                  <div className="flex-1">
                    <div className="text-sm font-medium">{t("privacy_policy")}</div>
                    <div className="text-xs text-muted-foreground mt-0.5">
                      {t("privacy_policy_note")}
                    </div>
                  </div>
                </button>
              )}
            </div>
          </section>
        </div>

        <div className="p-4 border-t border-border">
          <SheetClose asChild>
            <button className="w-full px-4 py-2.5 rounded-lg bg-muted hover:bg-muted/70 text-sm font-medium transition-colors">
              {t("close")}
            </button>
          </SheetClose>
        </div>
      </SheetContent>

      <Dialog open={privacyOpen} onOpenChange={setPrivacyOpen}>
        <DialogContent className="sm:max-w-lg max-h-[85vh] flex flex-col">
          <DialogHeader>
            <DialogTitle>{t("privacy_policy")}</DialogTitle>
            <DialogDescription>
              {t("privacy_policy_note")}
            </DialogDescription>
          </DialogHeader>

          <div className="flex-1 overflow-y-auto space-y-5 text-sm leading-6 text-foreground/90 pr-1">
            <section>
              <h4 className="font-medium mb-1.5 text-foreground">
                Medical disclaimer
              </h4>
              <p>
                DocuMed AI is a health navigator, not a doctor. It cannot
                diagnose, prescribe, or provide a treatment plan. Every
                answer is general information drawn from public-health
                sources and should be confirmed with a qualified clinician
                before acting on it.
              </p>
              <p className="mt-2">
                For any medical emergency — severe chest pain, stroke
                symptoms, heavy bleeding, difficulty breathing, a seizure,
                loss of consciousness, or a severe allergic reaction —
                call <strong>102</strong> (Nepal ambulance) or go to the
                nearest emergency department immediately. Do not wait on
                a chat answer.
              </p>
            </section>

            <section>
              <h4 className="font-medium mb-1.5 text-foreground">
                Data we store
              </h4>
              <p>
                When you sign in we store your email address, a display
                name if you provided one, the chats you have in this app,
                and any documents you upload (lab reports or research
                PDFs). Passwords are never stored in plain text — they
                are hashed by our authentication provider (Supabase).
              </p>
              <p className="mt-2">
                Uploaded PDFs are stored as extracted text plus the
                lab-marker values parsed from them, tied to your user
                account. Chat transcripts are stored per-session so you
                can re-open them later from the sidebar.
              </p>
            </section>

            <section>
              <h4 className="font-medium mb-1.5 text-foreground">
                Your data, your control
              </h4>
              <p>
                You can clear every saved chat from{" "}
                <em>Settings → Data & Privacy → Clear all chats</em>. You
                can delete your account and all associated data from{" "}
                <em>Settings → Data & Privacy → Delete my account</em>.
                Account deletion is irreversible and cascades across
                every table tied to your user id.
              </p>
            </section>

            <section>
              <h4 className="font-medium mb-1.5 text-foreground">
                Sources and citations
              </h4>
              <p>
                Answers are grounded in public-health references
                including NHS (UK), WHO fact sheets, MedlinePlus (US
                National Library of Medicine), and Nepal MoHP / DoHS
                publications. Citations are shown under each answer.
                DocuMed AI will refuse an answer rather than guess when
                its sources do not cover your question.
              </p>
            </section>

            <section>
              <h4 className="font-medium mb-1.5 text-foreground">
                Safety logging
              </h4>
              <p>
                We log a record of each query for safety auditing — which
                red-flag rules fired, how the answer was graded by the
                source-grounding check, and whether a refusal was
                emitted. This is used only to find and fix cases where
                the app might give an unsafe answer. Logs are tied to
                your user id and are removed when you delete your
                account.
              </p>
            </section>

            <section>
              <h4 className="font-medium mb-1.5 text-foreground">
                Questions or concerns
              </h4>
              <p>
                If something in an answer looks wrong or unsafe, stop,
                speak to a clinician, and tell us so we can correct it.
              </p>
            </section>
          </div>

          <DialogFooter className="mt-2">
            <button
              type="button"
              onClick={() => setPrivacyOpen(false)}
              className="w-full sm:w-auto px-4 py-2 rounded-lg bg-muted hover:bg-muted/70 text-sm font-medium transition-colors"
            >
              {t("close")}
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Sheet>
  );
}
