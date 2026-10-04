import type { Metadata } from "next";
import { Building2, HandCoins, KeyRound, LineChart, Ruler, Search } from "lucide-react";
import { CtaBar } from "@/components/CtaBar";
import { PageHero } from "@/components/PageHero";
import { Reveal } from "@/components/Reveal";
import { SectionHeading } from "@/components/SectionHeading";

export const metadata: Metadata = {
  title: "خدمات",
  description:
    "خدمات افق املاک: خرید و فروش، اجاره، مشاوره سرمایه‌گذاری، ارزیابی ملک و مدیریت املاک.",
  alternates: { canonical: "/services" },
};

const services = [
  { icon: Search, title: "خرید ملک", text: "جست‌وجوی هدفمند و بازدید از گزینه‌های منتخب تا رسیدن به خانه دلخواه." },
  { icon: KeyRound, title: "فروش ملک", text: "بازاریابی حرفه‌ای، عکاسی تخصصی و مذاکره برای رسیدن به بهترین قیمت." },
  { icon: HandCoins, title: "اجاره و رهن", text: "مدیریت کامل فرایند اجاره، از معرفی مستأجر تا تنظیم قرارداد." },
  { icon: LineChart, title: "مشاوره سرمایه‌گذاری", text: "تحلیل بازار و پیشنهاد فرصت‌های پرسود متناسب با بودجه شما." },
  { icon: Ruler, title: "ارزیابی ملک", text: "برآورد دقیق ارزش ملک بر اساس داده‌های واقعی بازار." },
  { icon: Building2, title: "مدیریت املاک", text: "نگهداری، اجاره و نظارت بر املاک شما به‌صورت کامل." },
];

export default function ServicesPage() {
  return (
    <>
      <PageHero
        eyebrow="خدمات"
        title="خدمات ما"
        description="از اولین جست‌وجو تا امضای قرارداد و پس از آن، تمام خدمات ملکی را زیر یک سقف دریافت کنید."
      />

      <section className="section bg-white">
        <div className="container-x">
          <Reveal>
            <SectionHeading
              eyebrow="چه می‌کنیم"
              title="راهکار کامل املاک"
              description="هر خدمت با تیمی متخصص و فرایندی شفاف ارائه می‌شود."
            />
          </Reveal>
          <div className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {services.map((s, i) => (
              <Reveal key={s.title} delay={i * 0.05}>
                <div className="h-full rounded-3xl bg-bg-soft p-7 shadow-card ring-1 ring-navy/5 transition hover:-translate-y-1 hover:shadow-card-hover">
                  <span className="grid h-12 w-12 place-items-center rounded-2xl bg-navy text-gold">
                    <s.icon className="h-6 w-6" aria-hidden />
                  </span>
                  <h3 className="mt-5 text-lg font-extrabold text-navy">{s.title}</h3>
                  <p className="mt-2 leading-7 text-muted">{s.text}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <CtaBar />
    </>
  );
}
