import Image from "next/image";
import Link from "next/link";
import { ArrowLeft, ChevronLeft } from "lucide-react";
import { Reveal } from "@/components/Reveal";
import { SectionHeading } from "@/components/SectionHeading";

export function About() {
  return (
    <section id="about" className="section bg-white">
      <div className="container-x grid items-center gap-12 lg:grid-cols-2 lg:gap-16">
        <Reveal>
          <SectionHeading align="start" eyebrow="درباره ما" title="ما چه کسانی هستیم؟" />
          <p className="mt-6 leading-8 text-muted">
            افق املاک با بیش از هجده سال تجربه در بازار املاک لوکس، به خانواده‌ها و
            سرمایه‌گذاران کمک می‌کند تا با اطمینان تصمیم بگیرند. ما به کیفیت، شفافیت و
            پیگیری تا پس از معامله باور داریم.
          </p>
          <p className="mt-4 leading-8 text-muted">
            تیم کارشناسان ما در هر گام همراه شماست؛ از انتخاب ملک و بازدید تا مذاکره،
            قرارداد و خدمات پس از فروش.
          </p>
          <Link href="/about" className="btn-navy mt-8">
            بیشتر بدانید
            <ArrowLeft className="h-4 w-4" aria-hidden />
          </Link>
        </Reveal>

        <Reveal delay={0.15}>
          <div className="flex items-stretch gap-4">
            <div className="relative aspect-square flex-1 overflow-hidden rounded-3xl shadow-card">
              <Image
                src="https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=1200&q=80"
                alt="ویلای مدرن دو طبقه با بالکن و چمن"
                fill
                sizes="(max-width:1024px) 100vw, 40vw"
                className="object-cover"
              />
            </div>
            <div className="relative hidden aspect-square w-1/3 overflow-hidden rounded-3xl shadow-card sm:block">
              <Image
                src="https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?auto=format&fit=crop&w=800&q=60"
                alt="نمای داخلی لوکس ویلا"
                fill
                sizes="20vw"
                className="scale-110 object-cover blur-[1px]"
              />
              <span className="absolute inset-0 grid place-items-center bg-navy/30">
                <span className="grid h-11 w-11 place-items-center rounded-full bg-white text-navy shadow-card">
                  <ChevronLeft className="h-5 w-5" aria-hidden />
                </span>
              </span>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
