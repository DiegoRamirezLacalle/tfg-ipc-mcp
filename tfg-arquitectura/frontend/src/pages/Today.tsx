import { motion } from "motion/react";
import { Radio } from "lucide-react";

import { NewsPulse } from "@/components/news/NewsPulse";

export default function Today() {
  return (
    <div className="flex flex-col gap-6">
      <motion.div
        className="flex flex-col gap-2 border-b border-border pb-5"
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <div className="flex items-center gap-3">
          <Radio size={20} className="text-gold" />
          <h1 className="font-sans text-display-lg tracking-tight text-foreground">Today's Inflation Pulse</h1>
          <span className="pill pill-mcp text-[10px] ml-1">◈ live</span>
        </div>
        <p className="font-mono text-data-sm text-foreground-muted max-w-2xl">
          Real-time inflation headlines pulled from <span className="text-gold">GDELT</span> and scored live by{" "}
          <span className="text-mcp">FinBERT</span> through the MCP server — the same sentiment pipeline that
          feeds the forecasting models' C1 context.
        </p>
      </motion.div>

      <NewsPulse limit={14} showRefresh />
    </div>
  );
}
