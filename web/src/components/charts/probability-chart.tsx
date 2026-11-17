"use client";

import {
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ModelProbabilities } from "@/lib/types";
import { ResponsiveChartContainer } from "@/components/charts/responsive-chart-container";

export function ProbabilityChart({
  probabilities,
}: {
  probabilities: ModelProbabilities | null;
}) {
  const data = probabilities
    ? [
        { name: "Control", value: probabilities.control, color: "#64748b" },
        { name: "ADHD", value: probabilities.adhd, color: "#635bdb" },
      ].filter((item) => Number.isFinite(item.value))
    : [];

  if (data.length === 0) {
    return (
      <div className="grid h-52 place-items-center rounded-xl bg-slate-50 text-sm text-slate-500">
        Probability output is not available.
      </div>
    );
  }

  return (
    <div
      role="img"
      aria-label="Model output probabilities for Control and ADHD"
      className="h-52 min-w-0 w-full"
    >
      <ResponsiveChartContainer>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 8, right: 24, bottom: 8, left: 4 }}
        >
          <CartesianGrid horizontal={false} stroke="#edf0f6" />
          <XAxis
            type="number"
            domain={[0, 1]}
            tickFormatter={(value: number) => `${Math.round(value * 100)}%`}
            tick={{ fill: "#64748b", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={68}
            tick={{ fill: "#334155", fontSize: 12, fontWeight: 600 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            formatter={(value) => [
              `${(Number(value) * 100).toFixed(1)}%`,
              "Model output",
            ]}
            contentStyle={{
              borderRadius: 12,
              border: "1px solid #e2e8f0",
              boxShadow: "0 8px 24px rgba(15, 23, 42, .08)",
            }}
          />
          <Bar dataKey="value" name="Model output probability" barSize={19} radius={7}>
            {data.map((item) => (
              <Cell key={item.name} fill={item.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveChartContainer>
    </div>
  );
}
