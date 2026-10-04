import Image from "next/image";
import Link from "next/link";
import { ChevronDown, Phone } from "lucide-react";
import { site } from "@/lib/site";

const HERO_IMAGE =
  "https://images.unsplash.com/photo-1613490493576-7fde63acd811?auto=format&fit=crop&w=2000&q=80";

export function Hero() {
  return (
    <section className="relative flex min-h-[100svh] items-center justify-center overflow-hidden">
      <Image
        src={HERO_IMAGE}
        alt="ویلای لوکس مدرن با استخر در غروب"
        fill
        priority
        sizes="100vw"
        className="object-cover"
      />
      <div className="absolute inset-0 bg-gradient-to-b from-navy-dark/85 via-navy/65 to-navy-dark/90" />

      <div className="container-x relative z-10 text-center text-white">
        <p className="eyebrow mb-4 animate-fade-up">افق املاک</p>
        <h1 className="mx-auto max-w-4xl animate-fade-up text-4xl font-extrabold leading-tight sm:text-5xl lg:text-6xl">
          خانه‌ها و سرمایه‌گذاری‌های استثنایی را کشف کنید
        </h1>
        <p className="mx-auto mt-6 max-w-2xl animate-fade-up text-base leading-8 text-white/85 sm:text-lg">
          املاک ممتاز در موقعیت‌های برتر. خانه رویایی یا سرمایه‌گذاری ایده‌آل خود را با
          اطمینان پیدا کنید.
        </p>
        <div className="mt-9 flex flex-wrap items-center justify-center gap-4">
          <Link href="/properties" className="btn bg-gold text-navy hover:bg-white">
            مشاهده املاک
          </Link>
          <a
            href={site.phoneHref}
            className="btn border border-white/40 text-white hover:bg-white hover:text-navy"
          >
            <Phone className="h-4 w-4" aria-hidden />
            تماس با ما
          </a>
        </div>
      </div>

      <a
        href="#about"
        aria-label="رفتن به بخش درباره ما"
        className="absolute bottom-6 left-1/2 z-10 -translate-x-1/2 animate-float text-white/70 hover:text-white"
      >
        <ChevronDown className="h-7 w-7" aria-hidden />
      </a>
    </section>
  );
}
