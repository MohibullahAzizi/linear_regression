"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Heart } from "lucide-react";
import { PropertyCard } from "@/components/PropertyCard";
import { FAVORITES_EVENT, getFavorites } from "@/lib/favorites";
import type { Property } from "@/lib/types";

export function FavoritesList({ all }: { all: Property[] }) {
  const [slugs, setSlugs] = useState<string[]>([]);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const sync = () => setSlugs(getFavorites());
    sync();
    setReady(true);
    window.addEventListener(FAVORITES_EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(FAVORITES_EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  const items = all.filter((p) => slugs.includes(p.slug));

  if (!ready) {
    return <p className="py-16 text-center text-muted">در حال بارگذاری…</p>;
  }

  if (items.length === 0) {
    return (
      <div className="rounded-3xl border border-dashed border-navy/15 p-12 text-center">
        <span className="mx-auto grid h-14 w-14 place-items-center rounded-full bg-navy/5 text-gold">
          <Heart className="h-7 w-7" aria-hidden />
        </span>
        <p className="mt-4 font-semibold text-navy">هنوز ملکی را به علاقه‌مندی‌ها اضافه نکرده‌اید.</p>
        <p className="mt-2 text-sm text-muted">
          روی آیکون قلب هر ملک بزنید تا اینجا ذخیره شود.
        </p>
        <Link href="/properties" className="btn-navy mt-6">
          مشاهده املاک
        </Link>
      </div>
    );
  }

  return (
    <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
      {items.map((p, i) => (
        <PropertyCard key={p.slug} property={p} priority={i < 3} />
      ))}
    </div>
  );
}
