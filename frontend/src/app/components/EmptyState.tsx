import type { ReactNode } from "react";
import { motion } from "motion/react";
import {
  Stethoscope,
  Brain,
  Heart,
  Microscope,
  Pill,
  Activity,
  FileText,
  Zap,
  Shield,
  TrendingUp,
  Sparkles,
  ChevronRight,
} from "lucide-react";

interface ExamplePrompt {
  icon: ReactNode;
  category: string;
  title: string;
  description: string;
  gradient: string;
}

interface EmptyStateProps {
  onPromptClick: (prompt: string) => void;
  inputElement?: ReactNode;
}

export function EmptyState({ onPromptClick, inputElement }: EmptyStateProps) {
  const examplePrompts: ExamplePrompt[] = [
    {
      icon: <Heart className="w-5 h-5" />,
      category: "Cardiology",
      title: "Cardiovascular Risk Assessment",
      description:
        "What are the latest guidelines for managing patients with heart failure and reduced ejection fraction?",
      gradient: "from-red-500/10 to-pink-500/10",
    },
    {
      icon: <Brain className="w-5 h-5" />,
      category: "Neurology",
      title: "Neurological Disorders",
      description:
        "Summarize recent advances in Alzheimer's disease treatment and early diagnostic markers.",
      gradient: "from-purple-500/10 to-indigo-500/10",
    },
    {
      icon: <Pill className="w-5 h-5" />,
      category: "Pharmacology",
      title: "Drug Interactions",
      description:
        "Analyze potential drug interactions between metformin, atorvastatin, and lisinopril.",
      gradient: "from-blue-500/10 to-cyan-500/10",
    },
    {
      icon: <Microscope className="w-5 h-5" />,
      category: "Oncology",
      title: "Cancer Research",
      description:
        "What are the current immunotherapy options for stage III non-small cell lung cancer?",
      gradient: "from-emerald-500/10 to-teal-500/10",
    },
  ];

  const capabilities = [
    {
      icon: <FileText className="w-5 h-5 text-accent" />,
      title: "Evidence-Based Insights",
      description: "Access to 2.8M+ peer-reviewed medical documents",
    },
    {
      icon: <Zap className="w-5 h-5 text-accent" />,
      title: "Real-Time Retrieval",
      description: "Sub-second query processing with RAG technology",
    },
    {
      icon: <Shield className="w-5 h-5 text-accent" />,
      title: "Clinical Accuracy",
      description: "96% average confidence score on medical queries",
    },
    {
      icon: <TrendingUp className="w-5 h-5 text-accent" />,
      title: "Latest Research",
      description: "Updated daily with newest clinical studies",
    },
  ];

  const stats = [
    { value: "2.8M+", label: "Research Papers" },
    { value: "98.7%", label: "Accuracy Rate" },
    { value: "1.2s", label: "Avg Response" },
    { value: "150K+", label: "Clinical Queries" },
  ];

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-12" style={{ marginTop: "-30px" }}>
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          {/* Animated medical icon cluster */}
          <div className="relative w-32 h-32 mx-auto mb-8">
            <motion.div
              animate={{
                rotate: 360,
              }}
              transition={{
                duration: 40,
                repeat: Infinity,
                ease: "linear",
              }}
              className="absolute inset-0"
            >
              {[Stethoscope, Heart, Brain, Microscope, Pill, Activity].map((Icon, idx) => {
                const angle = (idx * 360) / 6;
                const radius = 50;
                const x = Math.cos((angle * Math.PI) / 180) * radius;
                const y = Math.sin((angle * Math.PI) / 180) * radius;

                return (
                  <motion.div
                    key={idx}
                    className="absolute top-1/2 left-1/2 w-12 h-12 -ml-6 -mt-6 bg-accent/10 rounded-xl flex items-center justify-center border border-accent/20"
                    style={{
                      transform: `translate(${x}px, ${y}px)`,
                    }}
                    animate={{
                      rotate: -360,
                      scale: [1, 1.1, 1],
                    }}
                    transition={{
                      rotate: {
                        duration: 40,
                        repeat: Infinity,
                        ease: "linear",
                      },
                      scale: {
                        duration: 2,
                        repeat: Infinity,
                        delay: idx * 0.3,
                      },
                    }}
                  >
                    <Icon className="w-5 h-5 text-accent" />
                  </motion.div>
                );
              })}
            </motion.div>

            {/* Center pulse */}
            <motion.div
              className="absolute top-1/2 left-1/2 -ml-8 -mt-8 w-16 h-16 bg-accent rounded-2xl flex items-center justify-center"
              animate={{
                boxShadow: [
                  "0 0 0 0 rgba(0, 191, 165, 0.4)",
                  "0 0 0 20px rgba(0, 191, 165, 0)",
                ],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
              }}
            >
              <Sparkles className="w-7 h-7 text-white" />
            </motion.div>
          </div>

          <h1 className="text-4xl md:text-5xl font-medium mb-4 bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
            Medical Intelligence at Your Fingertips
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-8">
            Harness the power of advanced RAG technology to access evidence-based medical
            insights from millions of peer-reviewed sources. Ask anything, get precise
            clinical answers.
          </p>

          {/* Stats bar */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto mb-12">
            {stats.map((stat, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1 * idx }}
                className="bg-card border border-border rounded-xl p-4"
              >
                <div className="text-2xl font-mono font-medium text-accent mb-1">
                  {stat.value}
                </div>
                <div className="text-xs text-muted-foreground">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {inputElement && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="sticky top-4 z-20 mb-12"
          >
            <div className="py-1">
              {inputElement}
              <div className="text-center mt-3">
                <p className="text-xs text-muted-foreground font-mono">
                  Press <kbd className="px-2 py-0.5 bg-muted rounded border border-border">Enter</kbd> to send • <kbd className="px-2 py-0.5 bg-muted rounded border border-border">Shift+Enter</kbd> for new line
                </p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Example Prompts */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25, duration: 0.5 }}
          className="mb-16"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="h-px flex-1 bg-gradient-to-r from-transparent via-border to-transparent" />
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
              Explore Medical Topics
            </h2>
            <div className="h-px flex-1 bg-gradient-to-r from-transparent via-border to-transparent" />
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            {examplePrompts.map((prompt, idx) => (
              <motion.button
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + idx * 0.1 }}
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => onPromptClick(prompt.description)}
                className={`group relative bg-gradient-to-br ${prompt.gradient} border border-border rounded-2xl p-6 text-left overflow-hidden hover:border-accent/50 transition-all`}
              >
                {/* Background decoration */}
                <div className="absolute top-0 right-0 w-32 h-32 bg-accent/5 rounded-full blur-3xl" />

                <div className="relative">
                  <div className="flex items-start justify-between mb-4">
                    <div className="p-2.5 bg-card rounded-xl border border-border">
                      {prompt.icon}
                    </div>
                    <ChevronRight className="w-5 h-5 text-muted-foreground group-hover:text-accent group-hover:translate-x-1 transition-all" />
                  </div>

                  <div className="text-xs font-mono font-medium text-accent uppercase tracking-wider mb-2">
                    {prompt.category}
                  </div>
                  <h3 className="font-medium mb-2">{prompt.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {prompt.description}
                  </p>
                </div>
              </motion.button>
            ))}
          </div>
        </motion.div>

        {/* Capabilities */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          className="mb-12"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="h-px flex-1 bg-gradient-to-r from-transparent via-border to-transparent" />
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
              Advanced Capabilities
            </h2>
            <div className="h-px flex-1 bg-gradient-to-r from-transparent via-border to-transparent" />
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {capabilities.map((capability, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 + idx * 0.1 }}
                className="bg-card border border-border rounded-xl p-5 hover:border-accent/50 transition-colors"
              >
                <div className="mb-3">{capability.icon}</div>
                <h3 className="text-sm font-medium mb-1">{capability.title}</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {capability.description}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>

      </div>
    </div>
  );
}
