import { useEffect, useRef } from "react";
import { animate } from "motion";

interface Props {
  value: number;
  decimals?: number;
  suffix?: string;
  className?: string;
}

export function NumberTicker({ value, decimals = 3, suffix = "", className }: Props) {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const controls = animate(0, value, {
      duration: 1.2,
      ease: [0.16, 1, 0.3, 1],
      onUpdate(v: number) {
        el.textContent = v.toFixed(decimals) + suffix;
      },
    });
    return () => controls.stop();
  }, [value, decimals, suffix]);

  return (
    <span ref={ref} className={className}>
      {value.toFixed(decimals)}{suffix}
    </span>
  );
}
