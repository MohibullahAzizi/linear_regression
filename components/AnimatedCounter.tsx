"use client";

import { useEffect, useRef, useState } from "react";
import { toFa } from "@/lib/format";

type Props = {
  value: number;
  prefix?: string;
  suffix?: string;
  duration?: number;
};

/** Counts up from 0 to `value` when scrolled into view. */
export function AnimatedCounter({ value, prefix = "", suffix = "", duration = 1800 }: Props) {
  const ref = useRef<HTMLSpanElement>(null);
  const started = useRef(false);
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const io = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting || started.current) return;
        started.current = true;
        const start = performance.now();
        const tick = (now: number) => {
          const p = Math.min((now - start) / duration, 1);
          const eased = 1 - Math.pow(1 - p, 3);
          setDisplay(Math.round(eased * value));
          if (p < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      },
      { threshold: 0.4 },
    );

    io.observe(el);
    return () => io.disconnect();
  }, [value, duration]);

  return (
    <span ref={ref} aria-label={`${value}`}>
      {prefix}
      {toFa(display.toLocaleString("en-US"))}
      {suffix}
    </span>
  );
}
