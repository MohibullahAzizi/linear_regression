"use client";

import { Heart } from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/cn";
import { FAVORITES_EVENT, isFavorite, toggleFavorite } from "@/lib/favorites";

type Props = {
  slug: string;
  title: string;
  className?: string;
};

export function FavoriteButton({ slug, title, className }: Props) {
  const [active, setActive] = useState(false);

  useEffect(() => {
    setActive(isFavorite(slug));
    const onChange = () => setActive(isFavorite(slug));
    window.addEventListener(FAVORITES_EVENT, onChange);
    window.addEventListener("storage", onChange);
    return () => {
      window.removeEventListener(FAVORITES_EVENT, onChange);
      window.removeEventListener("storage", onChange);
    };
  }, [slug]);

  return (
    <button
      type="button"
      onClick={(e) => {
        e.preventDefault();
        setActive(toggleFavorite(slug).includes(slug));
      }}
      aria-pressed={active}
      aria-label={active ? `حذف ${title} از علاقه‌مندی‌ها` : `افزودن ${title} به علاقه‌مندی‌ها`}
      className={cn(
        "grid h-10 w-10 place-items-center rounded-full bg-white/90 text-navy shadow-card backdrop-blur",
        "transition hover:scale-110 hover:text-gold focus-visible:outline-none",
        active && "text-gold",
        className,
      )}
    >
      <Heart className={cn("h-5 w-5", active && "fill-current")} aria-hidden />
    </button>
  );
}
