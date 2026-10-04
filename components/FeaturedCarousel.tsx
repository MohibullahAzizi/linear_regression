"use client";

import { useRef } from "react";
import { Swiper, SwiperSlide } from "swiper/react";
import { A11y, Autoplay, Pagination } from "swiper/modules";
import type { Swiper as SwiperType } from "swiper";
import { ChevronLeft, ChevronRight } from "lucide-react";
import "swiper/css";
import "swiper/css/pagination";

import { PropertyCard } from "@/components/PropertyCard";
import type { Property } from "@/lib/types";

export function FeaturedCarousel({ items }: { items: Property[] }) {
  const swiperRef = useRef<SwiperType | null>(null);

  return (
    <div className="relative">
      <Swiper
        dir="rtl"
        modules={[Pagination, Autoplay, A11y]}
        spaceBetween={20}
        slidesPerView={1.12}
        grabCursor
        speed={600}
        autoplay={{ delay: 5000, disableOnInteraction: true }}
        pagination={{ clickable: true }}
        onSwiper={(s) => (swiperRef.current = s)}
        breakpoints={{
          640: { slidesPerView: 1.6, spaceBetween: 20 },
          768: { slidesPerView: 2.2, spaceBetween: 24 },
          1024: { slidesPerView: 3, spaceBetween: 24 },
          1280: { slidesPerView: 3.4, spaceBetween: 28 },
        }}
        className="!pb-16"
        a11y={{ enabled: true }}
      >
        {items.map((p, i) => (
          <SwiperSlide key={p.slug} className="!h-auto pb-2">
            <PropertyCard property={p} priority={i === 0} />
          </SwiperSlide>
        ))}
      </Swiper>

      {/* Arrows mirrored for RTL: prev points right, next points left */}
      <div className="mt-2 flex items-center justify-center gap-3">
        <button
          type="button"
          onClick={() => swiperRef.current?.slidePrev()}
          aria-label="اسلاید قبلی"
          className="grid h-11 w-11 place-items-center rounded-full bg-white text-navy shadow-card ring-1 ring-navy/10 transition hover:bg-navy hover:text-white"
        >
          <ChevronRight className="h-5 w-5" aria-hidden />
        </button>
        <button
          type="button"
          onClick={() => swiperRef.current?.slideNext()}
          aria-label="اسلاید بعدی"
          className="grid h-11 w-11 place-items-center rounded-full bg-white text-navy shadow-card ring-1 ring-navy/10 transition hover:bg-navy hover:text-white"
        >
          <ChevronLeft className="h-5 w-5" aria-hidden />
        </button>
      </div>
    </div>
  );
}
