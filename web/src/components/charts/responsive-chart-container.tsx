"use client";

import { useEffect, useState, type PropsWithChildren } from "react";
import { ResponsiveContainer } from "recharts";

export function ResponsiveChartContainer({ children }: PropsWithChildren) {
  const [resizeKey, setResizeKey] = useState(0);

  useEffect(() => {
    let resizeTimeout: ReturnType<typeof setTimeout> | undefined;

    function handleResize() {
      if (resizeTimeout) clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        setResizeKey((key) => key + 1);
      }, 100);
    }

    window.addEventListener("resize", handleResize, { passive: true });
    return () => {
      window.removeEventListener("resize", handleResize);
      if (resizeTimeout) clearTimeout(resizeTimeout);
    };
  }, []);

  return (
    <ResponsiveContainer
      key={resizeKey}
      width="100%"
      height="100%"
      minWidth={0}
      minHeight={0}
    >
      {children}
    </ResponsiveContainer>
  );
}
