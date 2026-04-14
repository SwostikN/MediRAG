import { Activity, FileText, TrendingUp, Database } from "lucide-react";

interface ContextMetric {
  label: string;
  value: string | number;
  change?: string;
  status?: "good" | "warning" | "critical";
}

interface MedicalContextPanelProps {
  metrics?: ContextMetric[];
  activeDocuments?: string[];
  ragStatus?: {
    vectorCount: number;
    lastUpdated: string;
    quality: number;
  };
}

export function MedicalContextPanel({
  metrics,
  activeDocuments,
  ragStatus,
}: MedicalContextPanelProps) {
  return (
    <div className="h-full bg-card border-l border-border overflow-y-auto">
      <div className="p-6 border-b border-border">
        <h3 className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-accent" />
          Medical Context
        </h3>
      </div>

      <div className="p-6 space-y-6">
        {metrics && metrics.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                Key Metrics
              </span>
            </div>
            <div className="space-y-3">
              {metrics.map((metric, idx) => (
                <div
                  key={idx}
                  className="bg-muted/30 rounded-lg p-4 border border-border/50"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="text-xs text-muted-foreground mb-1">
                        {metric.label}
                      </div>
                      <div className="text-2xl font-mono font-medium">{metric.value}</div>
                      {metric.change && (
                        <div
                          className="text-xs font-mono mt-1"
                          style={{
                            color:
                              metric.status === "good"
                                ? "#00BFA5"
                                : metric.status === "warning"
                                ? "#F59E0B"
                                : "#EF4444",
                          }}
                        >
                          {metric.change}
                        </div>
                      )}
                    </div>
                    {metric.status && (
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{
                          backgroundColor:
                            metric.status === "good"
                              ? "#00BFA5"
                              : metric.status === "warning"
                              ? "#F59E0B"
                              : "#EF4444",
                        }}
                      />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeDocuments && activeDocuments.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-4">
              <FileText className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                Active Documents
              </span>
            </div>
            <div className="space-y-2">
              {activeDocuments.map((doc, idx) => (
                <div
                  key={idx}
                  className="bg-muted/30 rounded-lg p-3 border border-border/50"
                >
                  <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-accent" />
                    <span className="text-sm font-medium">{doc}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {ragStatus && (
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Database className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                RAG Status
              </span>
            </div>
            <div className="bg-muted/30 rounded-lg p-4 border border-border/50 space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-xs text-muted-foreground">Vector Count</span>
                <span className="text-sm font-mono font-medium">
                  {ragStatus.vectorCount.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-muted-foreground">Quality Score</span>
                <span className="text-sm font-mono font-medium text-accent">
                  {Math.round(ragStatus.quality * 100)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-muted-foreground">Last Updated</span>
                <span className="text-xs font-mono">{ragStatus.lastUpdated}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
