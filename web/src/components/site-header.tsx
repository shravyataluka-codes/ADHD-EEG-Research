"use client";

import Link from "next/link";
import { Activity, LogIn, LogOut } from "lucide-react";
import { useAuth } from "@/lib/auth-context";

export function SiteHeader() {
  const { user, isAuthenticated, logout } = useAuth();

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
        <div className="flex items-center gap-3">
          {isAuthenticated && user ? (
            <div className="flex items-center gap-2.5">
              <div className="hidden rounded-full border border-indigo-100 bg-indigo-50/70 py-1 pl-1.5 pr-3 md:flex md:items-center md:gap-2">
                <span className="grid size-6 place-items-center rounded-full bg-indigo-700 text-[10px] font-bold text-white">
                  {user.name.charAt(0)}
                </span>
                <span className="text-xs font-semibold text-indigo-950">
                  {user.name}
                </span>
                <span className="rounded bg-indigo-200/70 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-indigo-800">
                  {user.role}
                </span>
              </div>
              <button
                type="button"
                onClick={logout}
                title="Sign out of research session"
                className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:border-rose-200 hover:bg-rose-50 hover:text-rose-700"
              >
                <LogOut size={13} />
                <span>Sign out</span>
              </button>
            </div>
          ) : (
            <Link
              href="/login"
              className="inline-flex items-center gap-1.5 rounded-full bg-indigo-700 px-4 py-2 text-xs font-semibold text-white shadow-sm transition hover:bg-indigo-800"
            >
              <LogIn size={14} />
              <span>Sign in</span>
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}
