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
import type { ChannelSlotAttributions } from "@/lib/types";
import { ResponsiveChartContainer } from "@/components/charts/responsive-chart-container";

interface RawAttributionChartProps {
  slot: 1 | 2;
  signed: ChannelSlotAttributions | null;
  absolute: ChannelSlotAttributions | null;
  samplingRateHz: number;
}

export function RawAttributionChart({
  slot,
  signed,
  absolute,
  samplingRateHz,
}: RawAttributionChartProps) {
  const key = slot === 1 ? "channel_slot_1" : "channel_slot_2";
  const signedValues = signed?.[key] ?? [];
  const absoluteValues = absolute?.[key] ?? [];
  const sampleCount = Math.max(signedValues.length, absoluteValues.length);
  const data = Array.from({ length: sampleCount }, (_, sample) => ({
    timeSeconds: sample / samplingRateHz,
    signed:
      Number.isFinite(signedValues[sample]) ? signedValues[sample] : null,
    absolute:
      Number.isFinite(absoluteValues[sample])
        ? absoluteValues[sample]
        : null,
  }));
  const hasData = data.some(
    (point) => point.signed !== null || point.absolute !== null,
  );

  if (!hasData || !Number.isFinite(samplingRateHz) || samplingRateHz <= 0) {
    return (
      <div className="grid h-64 place-items-center rounded-xl bg-slate-50 px-5 text-center text-sm text-slate-500">
        Raw attribution is unavailable for channel slot {slot}.
      </div>
    );
  }

  return (
    <div
      role="img"
      aria-label={`Mean signed and absolute raw EEG attribution by time for channel slot ${slot}`}
      className="h-64 min-w-0 w-full"
    >
      <ResponsiveChartContainer>
        <LineChart data={data} margin={{ top: 10, right: 10, bottom: 5, left: -12 }}>
          <CartesianGrid stroke="#edf0f6" vertical={false} />
          <XAxis
            dataKey="timeSeconds"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(value: number) => value.toFixed(1)}
            tick={{ fill: "#64748b", fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: "#d9deea" }}
            label={{
              value: "Epoch time (s)",
              position: "insideBottom",
              offset: -2,
              fill: "#64748b",
              fontSize: 10,
            }}
          />
          <YAxis
            tick={{ fill: "#64748b", fontSize: 10 }}
            tickFormatter={(value: number) => value.toExponential(1)}
            tickLine={false}
            axisLine={false}
            width={54}
            label={{
              value: "Attribution",
              angle: -90,
              position: "insideLeft",
              fill: "#64748b",
              fontSize: 10,
            }}
          />
          <Tooltip
            labelFormatter={(value) => `${Number(value).toFixed(3)} s`}
            formatter={(value, name) => [
              Number(value).toExponential(3),
              name === "signed" ? "Mean signed" : "Mean absolute",
            ]}
          />
          <Legend
            verticalAlign="top"
            align="right"
            iconType="plainline"
            wrapperStyle={{ fontSize: 11, paddingBottom: 6 }}
            formatter={(value) =>
              value === "signed" ? "Mean signed" : "Mean absolute"
            }
          />
          <Line
            type="monotone"
            dataKey="signed"
            name="signed"
            stroke="#635bdb"
            strokeWidth={1.7}
            dot={false}
            connectNulls={false}
          />
          <Line
            type="monotone"
            dataKey="absolute"
            name="absolute"
            stroke="#0f9b8e"
            strokeWidth={1.5}
            dot={false}
            connectNulls={false}
          />
        </LineChart>
      </ResponsiveChartContainer>
    </div>
  );
}
