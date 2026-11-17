import { AlertTriangle, ShieldCheck } from "lucide-react";

export function ResearchNotice({ compact = false }: { compact?: boolean }) {
  return (
    <aside
      className={`rounded-2xl border border-amber-200/80 bg-amber-50/80 text-amber-950 ${
        compact ? "p-4" : "p-5 sm:p-6"
      }`}
      aria-label="Research-use and scientific limitations"
    >
      <div className="flex items-start gap-3">
        <div className="mt-0.5 rounded-xl bg-amber-100 p-2 text-amber-800">
          {compact ? <ShieldCheck size={18} /> : <AlertTriangle size={19} />}
        </div>
        <div className="min-w-0">
          <h2 className="font-semibold tracking-tight">
            Exploratory research tool only
          </h2>
          <p className="mt-1 text-sm leading-6 text-amber-900/90">
            Not a clinical diagnostic system. Model probabilities are not
            calibrated confidence estimates, and attributions are not proof of
            causation.
          </p>
          {!compact && (
            <p className="mt-2 text-sm leading-6 text-amber-900/90">
              Participant identities have not been verified, and
              subject-independent generalization has not been established.
              Successful analysis does not imply clinically validated
              performance.
            </p>
          )}
        </div>
      </div>
    </aside>
  );
}
