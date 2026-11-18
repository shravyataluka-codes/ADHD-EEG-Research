"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  Eye,
  EyeOff,
  KeyRound,
  Lock,
  Mail,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { DEMO_CREDENTIALS, useAuth } from "@/lib/auth-context";

export default function LoginPage() {
  const router = useRouter();
  const { user, login } = useAuth();
  const [email, setEmail] = useState(DEMO_CREDENTIALS.email);
  const [password, setPassword] = useState(DEMO_CREDENTIALS.password);
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);

    try {
      const result = await login(email, password);
      if (result.success) {
        router.push("/");
      } else {
        setError(result.error || "Authentication failed. Please check your credentials.");
      }
    } catch {
      setError("An unexpected error occurred during sign-in.");
    } finally {
      setBusy(false);
    }
  }

  async function handleQuickDemo() {
    setEmail(DEMO_CREDENTIALS.email);
    setPassword(DEMO_CREDENTIALS.password);
    setError(null);
    setBusy(true);
    const result = await login(DEMO_CREDENTIALS.email, DEMO_CREDENTIALS.password);
    setBusy(false);
    if (result.success) {
      router.push("/");
    }
  }

  return (
    <div className="flex min-h-screen flex-col justify-between bg-gradient-to-br from-slate-50 via-indigo-50/30 to-slate-100">
      {/* Top Bar */}
      <header className="border-b border-slate-200/80 bg-white/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 sm:px-8">
          <Link
            className="flex items-center gap-3"
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
          <Link
            href="/"
            className="text-xs font-semibold text-slate-600 transition hover:text-indigo-700"
          >
            ← Back to workspace
          </Link>
        </div>
      </header>

      {/* Main Login Card */}
      <main className="mx-auto flex w-full max-w-md flex-1 flex-col justify-center px-4 py-12 sm:px-6">
        <div className="rounded-3xl border border-slate-200/90 bg-white p-7 shadow-[0_20px_50px_-25px_rgba(27,39,76,0.2)] sm:p-9">
          <div className="text-center">
            <div className="mx-auto inline-flex size-12 items-center justify-center rounded-2xl bg-indigo-50 text-indigo-700">
              <Lock size={22} />
            </div>
            <h1 className="mt-4 text-2xl font-bold tracking-tight text-slate-900">
              Sign in to Drushti Detect
            </h1>
            <p className="mt-1.5 text-xs text-slate-500">
              Access the clinical EEG explainability & analysis workspace
            </p>
          </div>

          {user && (
            <div className="mt-5 rounded-2xl border border-emerald-200 bg-emerald-50/70 p-3.5 text-xs text-emerald-900">
              <p className="font-semibold">Currently signed in as:</p>
              <p className="mt-0.5 text-emerald-800">
                {user.name} ({user.email})
              </p>
              <button
                type="button"
                onClick={() => router.push("/")}
                className="mt-2 inline-flex items-center gap-1 font-semibold text-emerald-950 underline hover:text-emerald-800"
              >
                Go to analysis workspace →
              </button>
            </div>
          )}

          {/* Quick Demo Button */}
          <div className="mt-6">
            <button
              type="button"
              onClick={handleQuickDemo}
              disabled={busy}
              className="group flex w-full items-center justify-between rounded-xl border border-indigo-200 bg-indigo-50/80 px-4 py-3 text-left transition hover:border-indigo-300 hover:bg-indigo-100/70"
            >
              <div className="flex items-center gap-2.5">
                <Sparkles size={17} className="text-indigo-600" />
                <div>
                  <span className="block text-xs font-bold text-indigo-950">
                    Quick Demo Sign-In
                  </span>
                  <span className="block text-[11px] text-indigo-700">
                    One-click access with verified researcher profile
                  </span>
                </div>
              </div>
              <ArrowRight
                size={16}
                className="text-indigo-500 transition-transform group-hover:translate-x-0.5"
              />
            </button>
          </div>

          <div className="relative my-6 flex items-center justify-center">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-slate-200" />
            </div>
            <span className="relative bg-white px-3 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
              or enter credentials
            </span>
          </div>

          {error && (
            <div
              role="alert"
              className="mb-5 rounded-xl border border-rose-200 bg-rose-50 p-3.5 text-xs text-rose-900"
            >
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label
                htmlFor="login-email"
                className="block text-xs font-semibold text-slate-700"
              >
                Institutional Email
              </label>
              <div className="relative mt-1.5 flex items-center">
                <Mail
                  size={16}
                  className="pointer-events-none absolute left-3.5 text-slate-400"
                />
                <input
                  id="login-email"
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="researcher@drushti.ai"
                  className="w-full rounded-xl border border-slate-200 bg-slate-50/50 py-2.5 pl-10 pr-3.5 text-sm text-slate-900 outline-none transition focus:border-indigo-500 focus:bg-white focus:ring-4 focus:ring-indigo-100"
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between">
                <label
                  htmlFor="login-password"
                  className="block text-xs font-semibold text-slate-700"
                >
                  Password
                </label>
                <span className="text-[11px] text-slate-400">
                  Demo: <code className="font-mono text-indigo-600 font-semibold">{DEMO_CREDENTIALS.password}</code>
                </span>
              </div>
              <div className="relative mt-1.5 flex items-center">
                <KeyRound
                  size={16}
                  className="pointer-events-none absolute left-3.5 text-slate-400"
                />
                <input
                  id="login-password"
                  type={showPassword ? "text" : "password"}
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••••••"
                  className="w-full rounded-xl border border-slate-200 bg-slate-50/50 py-2.5 pl-10 pr-10 text-sm text-slate-900 outline-none transition focus:border-indigo-500 focus:bg-white focus:ring-4 focus:ring-indigo-100"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 text-slate-400 transition hover:text-slate-600"
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between pt-1">
              <label className="flex items-center gap-2 text-xs text-slate-600">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  className="size-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                />
                Remember this session
              </label>
              <span className="text-xs text-indigo-600 hover:underline cursor-pointer">
                Forgot password?
              </span>
            </div>

            <button
              type="submit"
              disabled={busy}
              className="mt-2 flex w-full min-h-11 items-center justify-center gap-2 rounded-xl bg-indigo-700 px-4 py-2.5 text-sm font-semibold text-white shadow-md shadow-indigo-950/10 transition hover:bg-indigo-800 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {busy ? "Authenticating..." : "Sign in to workspace"}
            </button>
          </form>

          <div className="mt-6 rounded-2xl border border-slate-100 bg-slate-50 p-3.5 text-[11px] text-slate-500">
            <div className="flex items-center gap-2 text-slate-700 font-semibold">
              <ShieldCheck size={14} className="text-teal-600" />
              <span>Research Environment Access</span>
            </div>
            <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
              This system processes anonymized 2-channel EEG signals for ADHD vs Control research. Authorized investigator access only.
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200/80 bg-white/70 py-4 text-center text-xs text-slate-500">
        Drushti Detect • Exploratory EEG Research Workspace • Mendeley Dataset Benchmark
      </footer>
    </div>
  );
}
