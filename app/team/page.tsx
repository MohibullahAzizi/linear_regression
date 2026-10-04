import type { Metadata } from "next";
import { Mail, Phone } from "lucide-react";
import { CtaBar } from "@/components/CtaBar";
import { PageHero } from "@/components/PageHero";
import { Reveal } from "@/components/Reveal";
import { SectionHeading } from "@/components/SectionHeading";
import { site } from "@/lib/site";

export const metadata: Metadata = {
  title: "تیم ما",
  description:
    "با کارشناسان متخصص افق املاک آشنا شوید؛ تیمی از مشاوران املاک لوکس، مسکونی و تجاری.",
  alternates: { canonical: "/team" },
};

const team = [
  { name: "سارا محمدی", role: "مدیرعامل و مشاور ارشد", initial: "س" },
  { name: "امیر رضایی", role: "کارشناس پنت‌هاوس و برج", initial: "ا" },
  { name: "مریم کریمی", role: "مشاور املاک مسکونی", initial: "م" },
  { name: "رضا نیکزاد", role: "مشاور سرمایه‌گذاری ساحلی", initial: "ر" },
  { name: "نگار حسینی", role: "مشاور املاک مسکونی", initial: "ن" },
  { name: "کامران فرهادی", role: "مشاور املاک تجاری", initial: "ک" },
];

export default function TeamPage() {
  return (
    <>
      <PageHero
        eyebrow="تیم"
        title="با کارشناسان ما آشنا شوید"
        description="تیمی از مشاوران متخصص که بازار محلی را می‌شناسند و در کنار شما می‌مانند."
      />

      <section className="section bg-white">
        <div className="container-x">
          <Reveal>
            <SectionHeading eyebrow="افراد ما" title="متخصصان افق املاک" />
          </Reveal>
          <div className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {team.map((m, i) => (
              <Reveal key={m.name} delay={i * 0.05}>
                <div className="flex items-center gap-5 rounded-3xl bg-bg-soft p-6 shadow-card ring-1 ring-navy/5">
                  <span className="grid h-16 w-16 shrink-0 place-items-center rounded-full bg-navy text-2xl font-extrabold text-gold">
                    {m.initial}
                  </span>
                  <div>
                    <h3 className="font-extrabold text-navy">{m.name}</h3>
                    <p className="mt-1 text-sm text-muted">{m.role}</p>
                    <div className="mt-3 flex items-center gap-3 text-muted">
                      <a href={site.phoneHref} aria-label={`تماس با ${m.name}`} className="transition hover:text-gold">
                        <Phone className="h-4 w-4" aria-hidden />
                      </a>
                      <a href={`mailto:${site.email}`} aria-label={`ایمیل به ${m.name}`} className="transition hover:text-gold">
                        <Mail className="h-4 w-4" aria-hidden />
                      </a>
                    </div>
                  </div>
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
