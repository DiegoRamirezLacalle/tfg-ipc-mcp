import { motion } from "motion/react";

interface SectionCardProps {
  number: string;
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
  delay?: number;
  children: React.ReactNode;
}

export function SectionCard({ number, title, subtitle, right, delay = 0, children }: SectionCardProps) {
  return (
    <motion.section
      className="border-t border-border bg-card/40"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      <header className="flex items-end justify-between gap-4 px-4 pt-4 pb-3 border-b border-border">
        <div className="flex items-baseline gap-3">
          <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-foreground-subtle">
            {number}
          </span>
          <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-foreground-muted">
            {title}
          </span>
          {subtitle && (
            <span className="font-mono text-data-sm text-foreground-subtle hidden md:inline">
              {subtitle}
            </span>
          )}
        </div>
        {right}
      </header>
      <div className="p-4">{children}</div>
    </motion.section>
  );
}
