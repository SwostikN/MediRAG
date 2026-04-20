import { Globe, Type, Shield, Trash2, UserX, FileText } from "lucide-react";
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "./ui/sheet";
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

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-md flex flex-col">
        <SheetHeader className="text-left">
          <SheetTitle>{t("settings")}</SheetTitle>
          <SheetDescription>
            {t("answer_language_note")}
          </SheetDescription>
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

              <a
                href={privacyUrl ?? "/privacy"}
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
    </Sheet>
  );
}
