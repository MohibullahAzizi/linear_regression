"use client";

import { useMemo, useState } from "react";
import { Calculator } from "lucide-react";
import { formatToman, toFa } from "@/lib/format";

export function MortgageCalculator({ price }: { price: number }) {
  const [downPct, setDownPct] = useState(30);
  const [rate, setRate] = useState(18);
  const [years, setYears] = useState(10);

  const down = Math.round((price * downPct) / 100);
  const principal = Math.max(price - down, 0);

  const { monthly, total, totalInterest } = useMemo(() => {
    const r = rate / 100 / 12;
    const n = years * 12;
    const m = r === 0 ? principal / n : (principal * r) / (1 - Math.pow(1 + r, -n));
    const t = m * n;
    return { monthly: m, total: t, totalInterest: t - principal };
  }, [principal, rate, years]);

  const slider =
    "mt-2 h-2 w-full cursor-pointer appearance-none rounded-full bg-navy/10 accent-gold";

  return (
    <div className="rounded-3xl bg-white p-6 shadow-card ring-1 ring-navy/5">
      <div className="flex items-center gap-2">
        <span className="grid h-10 w-10 place-items-center rounded-full bg-navy/5 text-navy">
          <Calculator className="h-5 w-5" aria-hidden />
        </span>
        <h3 className="text-lg font-extrabold text-navy">ماشین‌حساب اقساط</h3>
      </div>

      <div className="mt-6 space-y-5">
        <div>
          <div className="flex items-center justify-between text-sm">
            <label htmlFor="mc-down" className="font-semibold text-navy">
              پیش‌پرداخت
            </label>
            <span className="text-muted">
              {toFa(downPct)}٪ · {formatToman(down)}
            </span>
          </div>
          <input
            id="mc-down"
            type="range"
            min={0}
            max={80}
            step={5}
            value={downPct}
            onChange={(e) => setDownPct(Number(e.target.value))}
            className={slider}
          />
        </div>

        <div>
          <div className="flex items-center justify-between text-sm">
            <label htmlFor="mc-rate" className="font-semibold text-navy">
              نرخ سود سالانه
            </label>
            <span className="text-muted">{toFa(rate)}٪</span>
          </div>
          <input
            id="mc-rate"
            type="range"
            min={0}
            max={30}
            step={0.5}
            value={rate}
            onChange={(e) => setRate(Number(e.target.value))}
            className={slider}
          />
        </div>

        <div>
          <div className="flex items-center justify-between text-sm">
            <label htmlFor="mc-years" className="font-semibold text-navy">
              مدت بازپرداخت
            </label>
            <span className="text-muted">{toFa(years)} سال</span>
          </div>
          <input
            id="mc-years"
            type="range"
            min={1}
            max={30}
            step={1}
            value={years}
            onChange={(e) => setYears(Number(e.target.value))}
            className={slider}
          />
        </div>
      </div>

      <dl className="mt-6 space-y-2 border-t border-navy/10 pt-5 text-sm">
        <div className="flex items-center justify-between">
          <dt className="text-muted">مبلغ وام</dt>
          <dd className="font-semibold text-text">{formatToman(principal)}</dd>
        </div>
        <div className="flex items-center justify-between">
          <dt className="text-muted">مجموع سود</dt>
          <dd className="font-semibold text-text">{formatToman(totalInterest)}</dd>
        </div>
        <div className="flex items-center justify-between">
          <dt className="text-muted">مجموع بازپرداخت</dt>
          <dd className="font-semibold text-text">{formatToman(total)}</dd>
        </div>
      </dl>

      <div className="mt-5 rounded-2xl bg-navy p-5 text-center text-white">
        <p className="text-xs text-white/70">قسط ماهانه</p>
        <p className="mt-1 text-2xl font-extrabold text-gold">{formatToman(monthly)}</p>
      </div>
    </div>
  );
}
