"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { ChevronLeft, ChevronRight, RotateCcw, Search } from "lucide-react";
import { PropertyCard } from "@/components/PropertyCard";
import { formatNumberFa, toFa } from "@/lib/format";
import {
  filterAndSort,
  getCities,
  propertyTypes,
  sortOptions,
  type SortKey,
} from "@/lib/properties";
import type { Property, PropertyType } from "@/lib/types";

const PAGE_SIZE = 6;

const priceRanges = [
  { label: "همه قیمت‌ها", min: 0, max: 0 },
  { label: "تا ۵۰۰ میلیون", min: 0, max: 500_000_000 },
  { label: "۵۰۰ میلیون تا ۱ میلیارد", min: 500_000_000, max: 1_000_000_000 },
  { label: "۱ تا ۳ میلیارد", min: 1_000_000_000, max: 3_000_000_000 },
  { label: "بیش از ۳ میلیارد", min: 3_000_000_000, max: 0 },
];

const selectClass =
  "w-full rounded-xl border border-navy/15 bg-white px-3 py-2.5 text-sm text-text focus:border-gold focus:outline-none focus:ring-2 focus:ring-gold/30";

export function PropertiesExplorer({ all }: { all: Property[] }) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();

  const get = (k: string, d = "") => params.get(k) ?? d;
  const q = get("q");
  const type = get("type", "all") as PropertyType | "all";
  const city = get("city", "all");
  const beds = Number(get("beds", "0"));
  const minPrice = Number(get("min", "0"));
  const maxPrice = Number(get("max", "0"));
  const sort = get("sort", "newest") as SortKey;
  const page = Math.max(1, Number(get("page", "1")));

  const update = (patch: Record<string, string | number | undefined>) => {
    const next = new URLSearchParams(params.toString());
    Object.entries(patch).forEach(([k, v]) => {
      if (v === undefined || v === "" || v === "all" || v === 0) next.delete(k);
      else next.set(k, String(v));
    });
    if (!("page" in patch)) next.delete("page");
    const qs = next.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  };

  const filtered = filterAndSort(
    all,
    { q, type, city, minPrice: minPrice || undefined, maxPrice: maxPrice || undefined, bedrooms: beds || undefined },
    sort,
  );

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const current = Math.min(page, totalPages);
  const pageItems = filtered.slice((current - 1) * PAGE_SIZE, current * PAGE_SIZE);

  const hasFilters =
    q !== "" || type !== "all" || city !== "all" || beds !== 0 || minPrice !== 0 || maxPrice !== 0 || sort !== "newest";

  const reset = () => router.replace(pathname, { scroll: false });

  return (
    <div>
      {/* Filters */}
      <div className="rounded-3xl bg-bg-soft p-5 shadow-card ring-1 ring-navy/5 sm:p-6">
        <div className="grid gap-4 lg:grid-cols-12">
          <div className="lg:col-span-4">
            <label htmlFor="f-q" className="sr-only">
              جست‌وجو
            </label>
            <div className="relative">
              <Search className="pointer-events-none absolute end-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" aria-hidden />
              <input
                id="f-q"
                defaultValue={q}
                onChange={(e) => update({ q: e.target.value })}
                placeholder="جست‌وجو در عنوان یا موقعیت..."
                className={`${selectClass} pe-9`}
              />
            </div>
          </div>

          <div className="lg:col-span-2">
            <label htmlFor="f-type" className="sr-only">نوع ملک</label>
            <select id="f-type" value={type} onChange={(e) => update({ type: e.target.value })} className={selectClass}>
              <option value="all">همه انواع</option>
              {propertyTypes.map((t) => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
          </div>

          <div className="lg:col-span-2">
            <label htmlFor="f-city" className="sr-only">شهر</label>
            <select id="f-city" value={city} onChange={(e) => update({ city: e.target.value })} className={selectClass}>
              <option value="all">همه شهرها</option>
              {getCities().map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

          <div className="lg:col-span-2">
            <label htmlFor="f-beds" className="sr-only">حداقل خواب</label>
            <select id="f-beds" value={beds} onChange={(e) => update({ beds: Number(e.target.value) })} className={selectClass}>
              <option value={0}>هر تعداد خواب</option>
              {[1, 2, 3, 4, 5].map((n) => (
                <option key={n} value={n}>{toFa(n)}+ خواب</option>
              ))}
            </select>
          </div>

          <div className="lg:col-span-2">
            <label htmlFor="f-price" className="sr-only">محدوده قیمت</label>
            <select
              id="f-price"
              value={`${minPrice}-${maxPrice}`}
              onChange={(e) => {
                const [min, max] = e.target.value.split("-").map(Number);
                update({ min: min || undefined, max: max || undefined });
              }}
              className={selectClass}
            >
              {priceRanges.map((r) => (
                <option key={r.label} value={`${r.min}-${r.max}`}>{r.label}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-muted">
            <span className="font-bold text-navy">{formatNumberFa(filtered.length)}</span> ملک یافت شد
          </p>
          <div className="flex items-center gap-2">
            {hasFilters && (
              <button type="button" onClick={reset} className="btn-outline-navy !px-4 !py-2 text-xs">
                <RotateCcw className="h-3.5 w-3.5" aria-hidden />
                حذف فیلترها
              </button>
            )}
            <label htmlFor="f-sort" className="sr-only">ترتیب</label>
            <select id="f-sort" value={sort} onChange={(e) => update({ sort: e.target.value })} className={`${selectClass} w-auto`}>
              {sortOptions.map((o) => (
                <option key={o.value} value={o.value}>ترتیب: {o.label}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Results */}
      {pageItems.length === 0 ? (
        <div className="mt-10 rounded-3xl border border-dashed border-navy/15 p-12 text-center">
          <p className="font-semibold text-navy">ملکی با این مشخصات پیدا نشد.</p>
          <p className="mt-2 text-sm text-muted">فیلترها را تغییر دهید یا جست‌وجو را پاک کنید.</p>
        </div>
      ) : (
        <div className="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {pageItems.map((p, i) => (
            <PropertyCard key={p.slug} property={p} priority={i < 3} />
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <nav className="mt-10 flex items-center justify-center gap-2" aria-label="صفحه‌بندی">
          <button
            type="button"
            onClick={() => update({ page: Math.max(1, current - 1) })}
            disabled={current === 1}
            aria-label="صفحه قبلی"
            className="grid h-10 w-10 place-items-center rounded-full bg-white text-navy shadow-card ring-1 ring-navy/10 transition hover:bg-navy hover:text-white disabled:opacity-40 disabled:hover:bg-white disabled:hover:text-navy"
          >
            <ChevronRight className="h-5 w-5" aria-hidden />
          </button>

          {Array.from({ length: totalPages }, (_, i) => i + 1).map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => update({ page: n })}
              aria-current={n === current ? "page" : undefined}
              className={
                n === current
                  ? "grid h-10 w-10 place-items-center rounded-full bg-navy text-sm font-bold text-white"
                  : "grid h-10 w-10 place-items-center rounded-full bg-white text-sm font-bold text-navy shadow-card ring-1 ring-navy/10 transition hover:bg-navy hover:text-white"
              }
            >
              {toFa(n)}
            </button>
          ))}

          <button
            type="button"
            onClick={() => update({ page: Math.min(totalPages, current + 1) })}
            disabled={current === totalPages}
            aria-label="صفحه بعدی"
            className="grid h-10 w-10 place-items-center rounded-full bg-white text-navy shadow-card ring-1 ring-navy/10 transition hover:bg-navy hover:text-white disabled:opacity-40 disabled:hover:bg-white disabled:hover:text-navy"
          >
            <ChevronLeft className="h-5 w-5" aria-hidden />
          </button>
        </nav>
      )}
    </div>
  );
}
