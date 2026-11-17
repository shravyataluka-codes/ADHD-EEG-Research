import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Drushti Detect | EEG Research",
  description:
    "Drushti Detect is an exploratory research interface for two-channel EEG-based ADHD vs Control analysis.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
