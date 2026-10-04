import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft, Award, Handshake, ShieldCheck, Target } from "lucide-react";
import { CtaBar } from "@/components/CtaBar";
import { PageHero } from "@/components/PageHero";
import { Reveal } from "@/components/Reveal";
import { SectionHeading } from "@/components/SectionHeading";
import { StatsCounter } from "@/components/StatsCounter";

export const metadata: Metadata = {
  title: "درباره ما",
  description:
    "افق املاک، مشاور املاک لوکس با هجده سال تجربه. داستان ما، ارزش‌ها و تعهد ما به شفافیت و کیفیت.",
  alternates: { canonical: "/about" },
};

const values = [
  { icon: ShieldCheck, title: "شفافیت", text: "همه اطلاعات ملک و معامله، بدون ابهام و پنهان‌کاری در اختیار شماست." },
  { icon: Award, title: "کیفیت", text: "تنها املاکی را عرضه می‌کنیم که خودمان به کیفیت و موقعیت آن‌ها اطمینان داریم." },
  { icon: Handshake, title: "تعهد", text: "از بازدید تا پس از معامله، همراه شما می‌مانیم." },
  { icon: Target, title: "دقت", text: "با تحلیل بازار، بهترین گزینه متناسب با هدف شما را پیشنهاد می‌دهیم." },
];

export default function AboutPage() {
  return (
    <>
      <PageHero
        eyebrow="درباره ما"
        title="ما چه کسانی هستیم؟"
        description="افق املاک از سال ۱۳۸۵ در بازار املاک لوکس فعالیت می‌کند و به بیش از هزار خانواده و سرمایه‌گذار خدمت رسانده است."
      />

      <section className="section bg-white">
        <div className="container-x grid items-center gap-12 lg:grid-cols-2">
          <Reveal>
            <SectionHeading align="start" title="داستان ما" />
            <p className="mt-6 leading-8 text-muted">
              افق املاک با هدف ساده‌ای آغاز شد: خرید و فروش ملک باید شفاف، حرفه‌ای و
              بدون استرس باشد. امروز با تیمی از کارشناسان متخصص در حوزه‌های مسکونی،
              تجاری و سرمایه‌گذاری، این مسیر را ادامه می‌دهیم.
            </p>
            <p className="mt-4 leading-8 text-muted">
              ما به رابطه بلندمدت با مشتریان باور داریم؛ به همین دلیل بسیاری از
              معاملات ما از طریق معرفی مشتریان قبلی انجام می‌شود.
            </p>
            <Link href="/contact" className="btn-navy mt-8">
              با ما در تماس باشید
              <ArrowLeft className="h-4 w-4" aria-hidden />
            </Link>
          </Reveal>

          <Reveal delay={0.15}>
            <div className="grid grid-cols-2 gap-4">
              {values.map((v) => (
                <div key={v.title} className="rounded-3xl bg-bg-soft p-6 shadow-card ring-1 ring-navy/5">
                  <span className="grid h-11 w-11 place-items-center rounded-full bg-navy text-gold">
                    <v.icon className="h-5 w-5" aria-hidden />
                  </span>
                  <h3 className="mt-4 font-bold text-navy">{v.title}</h3>
                  <p className="mt-2 text-sm leading-7 text-muted">{v.text}</p>
                </div>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      <StatsCounter />
      <CtaBar />
    </>
  );
}
