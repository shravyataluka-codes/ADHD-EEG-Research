"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FeatureAttribution } from "@/lib/types";
import { ResponsiveChartContainer } from "@/components/charts/responsive-chart-container";

export function FeatureAttributionChart({
  attributions,
}: {
  attributions: FeatureAttribution[];
}) {
  const data = [...attributions]
    .filter(
      (item) =>
        item.feature_name &&
        Number.isFinite(item.signed_contribution) &&
        Number.isFinite(item.absolute_magnitude),
    )
    .sort((left, right) => right.absolute_magnitude - left.absolute_magnitude)
    .slice(0, 10);

  if (data.length === 0) {
    return (
      <div className="grid h-72 place-items-center rounded-xl bg-slate-50 px-5 text-center text-sm text-slate-500">
        Feature attributions are unavailable for this recording.
      </div>
    );
  }

  return (
    <div
      role="img"
      aria-label="Top ten handcrafted feature attributions, ordered by absolute magnitude; signed bars show direction"
      className="h-[23rem] min-w-0 w-full"
    >
      <ResponsiveChartContainer>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 24, bottom: 10, left: 12 }}
        >
          <CartesianGrid horizontal={false} stroke="#edf0f6" />
          <XAxis
            type="number"
            tick={{ fill: "#64748b", fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: "#d9deea" }}
            label={{
              value: "Signed contribution to ADHD-vs-Control logit margin",
              position: "insideBottom",
              offset: -4,
              fill: "#64748b",
              fontSize: 10,
            }}
          />
          <YAxis
            type="category"
            dataKey="feature_name"
            width={125}
            tick={{ fill: "#475569", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
          />
          <ReferenceLine x={0} stroke="#94a3b8" />
          <Tooltip
            formatter={(value) => [
              Number(value).toExponential(3),
              "Signed contribution",
            ]}
            labelFormatter={(label) => String(label)}
            contentStyle={{
              borderRadius: 12,
              border: "1px solid #e2e8f0",
              boxShadow: "0 8px 24px rgba(15, 23, 42, .08)",
            }}
          />
          <Bar
            dataKey="signed_contribution"
            name="Signed contribution"
            barSize={15}
            radius={5}
          >
            {data.map((item) => (
              <Cell
                key={item.feature_name}
                fill={
                  item.signed_contribution >= 0 ? "#6558d9" : "#df7c69"
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveChartContainer>
    </div>
  );
}
