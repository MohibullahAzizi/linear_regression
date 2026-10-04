"use client";

import Image from "next/image";
import { useCallback, useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, X } from "lucide-react";

type Props = {
  images: string[];
  title: string;
};

export function PropertyGallery({ images, title }: Props) {
  const [index, setIndex] = useState<number | null>(null);

  const close = useCallback(() => setIndex(null), []);
  const prev = useCallback(
    () => setIndex((i) => (i === null ? i : (i - 1 + images.length) % images.length)),
    [images.length],
  );
  const next = useCallback(
    () => setIndex((i) => (i === null ? i : (i + 1) % images.length)),
    [images.length],
  );

  useEffect(() => {
    if (index === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
      if (e.key === "ArrowLeft") next(); // RTL: left points to the next image
      if (e.key === "ArrowRight") prev();
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [index, close, prev, next]);

  return (
    <>
      <div className="grid grid-cols-4 gap-3 sm:gap-4">
        <button
          type="button"
          onClick={() => setIndex(0)}
          aria-label="بزرگ‌نمایی تصویر اصلی"
          className="group relative col-span-4 aspect-[16/10] overflow-hidden rounded-3xl"
        >
          <Image
            src={images[0]}
            alt={`${title} — تصویر اصلی`}
            fill
            priority
            sizes="(max-width:1024px) 100vw, 66vw"
            className="object-cover transition duration-700 group-hover:scale-105"
          />
        </button>
        {images.slice(1, 5).map((src, i) => (
          <button
            key={`${src}-${i}`}
            type="button"
            onClick={() => setIndex(i + 1)}
            aria-label={`بزرگ‌نمایی تصویر ${i + 2}`}
            className="group relative col-span-2 aspect-[4/3] overflow-hidden rounded-2xl sm:col-span-1"
          >
            <Image
              src={src}
              alt={`${title} — تصویر ${i + 2}`}
              fill
              sizes="25vw"
              className="object-cover transition duration-700 group-hover:scale-105"
            />
          </button>
        ))}
      </div>

      {index !== null && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={`گالری تصاویر ${title}`}
          className="fixed inset-0 z-[60] flex items-center justify-center bg-navy-dark/95 p-4"
        >
          <button
            type="button"
            onClick={close}
            aria-label="بستن گالری"
            className="absolute end-4 top-4 grid h-11 w-11 place-items-center rounded-full bg-white/10 text-white transition hover:bg-white/20"
          >
            <X className="h-6 w-6" aria-hidden />
          </button>

          <button
            type="button"
            onClick={prev}
            aria-label="تصویر قبلی"
            className="absolute right-3 grid h-11 w-11 place-items-center rounded-full bg-white/10 text-white transition hover:bg-white/20 sm:right-6"
          >
            <ChevronRight className="h-6 w-6" aria-hidden />
          </button>

          <div className="relative h-[70vh] w-full max-w-4xl">
            <Image
              src={images[index]}
              alt={`${title} — تصویر ${index + 1}`}
              fill
              sizes="100vw"
              className="object-contain"
            />
          </div>

          <button
            type="button"
            onClick={next}
            aria-label="تصویر بعدی"
            className="absolute left-3 grid h-11 w-11 place-items-center rounded-full bg-white/10 text-white transition hover:bg-white/20 sm:left-6"
          >
            <ChevronLeft className="h-6 w-6" aria-hidden />
          </button>

          <p className="absolute bottom-6 text-sm text-white/80">
            {new Intl.NumberFormat("fa-IR").format(index + 1)} / {new Intl.NumberFormat("fa-IR").format(images.length)}
          </p>
        </div>
      )}
    </>
  );
}
