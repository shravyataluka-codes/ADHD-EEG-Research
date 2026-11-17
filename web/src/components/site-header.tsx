import Link from "next/link";
import { Activity, CircleHelp } from "lucide-react";

export function SiteHeader() {
  return (
    <header className="border-b border-slate-200/80 bg-white/75 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 sm:px-8">
        <Link
          className="flex items-center gap-3 rounded-lg"
          href="/"
          aria-label="Drushti Detect home"
        >
          <span className="grid size-10 place-items-center rounded-xl bg-indigo-950 text-white shadow-sm">
            <Activity size={21} strokeWidth={2.2} />
          </span>
          <span>
            <span className="block text-[15px] font-bold tracking-tight text-slate-900">
              Drushti Detect
            </span>
            <span className="block text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              EEG research workspace
            </span>
          </span>
        </Link>
        <a
          href="#about"
          className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3.5 py-2 text-sm font-medium text-slate-600 transition hover:border-indigo-200 hover:text-indigo-700"
        >
          <CircleHelp size={16} aria-hidden="true" />
          <span className="hidden sm:inline">About this tool</span>
        </a>
      </div>
    </header>
  );
}
