"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EpochAnalysis } from "@/lib/types";
import { ResponsiveChartContainer } from "@/components/charts/responsive-chart-container";

export function EpochProbabilityChart({
  epochs,
}: {
  epochs: EpochAnalysis[];
}) {
  const data = epochs
    .map((epoch) => ({
      epoch: epoch.epoch_index,
      control: epoch.model_output_probabilities?.control,
      adhd: epoch.model_output_probabilities?.adhd,
    }))
    .filter(
      (item) =>
        Number.isFinite(item.control) && Number.isFinite(item.adhd),
    );

  if (data.length === 0) {
    return (
      <div className="grid h-64 place-items-center rounded-xl bg-slate-50 px-5 text-center text-sm text-slate-500">
        Epoch probability chart is unavailable because there are no accepted
        epochs with probability outputs.
      </div>
    );
  }

  return (
    <div
      role="img"
      aria-label="Control and ADHD model output probabilities by accepted epoch"
      className="h-64 min-w-0 w-full"
    >
      <ResponsiveChartContainer>
        <LineChart data={data} margin={{ top: 10, right: 12, bottom: 2, left: -16 }}>
          <CartesianGrid stroke="#edf0f6" vertical={false} />
          <XAxis
            dataKey="epoch"
            type="number"
            domain={["dataMin", "dataMax"]}
            allowDecimals={false}
            tick={{ fill: "#64748b", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#d9deea" }}
            label={{
              value: "Accepted epoch",
              position: "insideBottom",
              offset: -2,
              fill: "#64748b",
              fontSize: 11,
            }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(value: number) => `${Math.round(value * 100)}%`}
            tick={{ fill: "#64748b", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            formatter={(value, name) => [
              `${(Number(value) * 100).toFixed(1)}%`,
              name === "control" ? "Control" : "ADHD",
            ]}
            labelFormatter={(value) => `Epoch ${value}`}
          />
          <Legend
            verticalAlign="top"
            align="right"
            iconType="circle"
            wrapperStyle={{ fontSize: 12, paddingBottom: 8 }}
            formatter={(value) => (value === "control" ? "Control" : "ADHD")}
          />
          <Line
            type="monotone"
            dataKey="control"
            name="control"
            stroke="#64748b"
            strokeWidth={2}
            dot={{ r: 3, fill: "#64748b" }}
            activeDot={{ r: 5 }}
          />
          <Line
            type="monotone"
            dataKey="adhd"
            name="adhd"
            stroke="#635bdb"
            strokeWidth={2.5}
            dot={{ r: 3, fill: "#635bdb" }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveChartContainer>
    </div>
  );
}
