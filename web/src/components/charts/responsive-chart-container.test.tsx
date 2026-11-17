import { act, render, screen } from "@testing-library/react";
import { useEffect, type PropsWithChildren } from "react";
import { describe, expect, it, vi } from "vitest";
import { ResponsiveChartContainer } from "@/components/charts/responsive-chart-container";

vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: PropsWithChildren) => (
    <div data-testid="responsive-container">{children}</div>
  ),
}));

function MountProbe({ onMount }: { onMount: () => void }) {
  useEffect(() => {
    onMount();
  }, [onMount]);

  return <div>Chart contents</div>;
}

describe("ResponsiveChartContainer", () => {
  it("recreates its Recharts container after a viewport resize settles", () => {
    vi.useFakeTimers();
    const onMount = vi.fn();

    try {
      render(
        <ResponsiveChartContainer>
          <MountProbe onMount={onMount} />
        </ResponsiveChartContainer>,
      );

      expect(onMount).toHaveBeenCalledTimes(1);
      expect(screen.getByText("Chart contents")).toBeInTheDocument();

      act(() => {
        window.dispatchEvent(new Event("resize"));
        vi.advanceTimersByTime(100);
      });

      expect(onMount).toHaveBeenCalledTimes(2);
      expect(screen.getByText("Chart contents")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });
});
