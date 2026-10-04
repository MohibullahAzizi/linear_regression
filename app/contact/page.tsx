import type { Metadata } from "next";
import { Clock, Mail, MapPin, Phone } from "lucide-react";
import { ContactForm } from "@/components/ContactForm";
import { MapEmbed } from "@/components/MapEmbed";
import { PageHero } from "@/components/PageHero";
import { Reveal } from "@/components/Reveal";
import { SectionHeading } from "@/components/SectionHeading";
import { site } from "@/lib/site";

export const metadata: Metadata = {
  title: "تماس با ما",
  description:
    "برای مشاوره خرید، فروش یا سرمایه‌گذاری ملکی با افق املاک تماس بگیرید. تلفن، ایمیل و آدرس دفتر.",
  alternates: { canonical: "/contact" },
};

export default function ContactPage() {
  return (
    <>
      <PageHero
        eyebrow="تماس"
        title="با ما در تماس باشید"
        description="کارشناسان ما آماده پاسخ‌گویی به سؤالات شما درباره خرید، فروش و سرمایه‌گذاری ملکی هستند."
      />

      <section className="section bg-white">
        <div className="container-x grid gap-12 lg:grid-cols-2 lg:gap-16">
          <Reveal>
            <SectionHeading align="start" eyebrow="فرم تماس" title="پیام خود را بفرستید" />
            <div className="mt-8">
              <ContactForm />
            </div>
          </Reveal>

          <Reveal delay={0.15}>
            <div className="space-y-6">
              <ul className="space-y-4 rounded-3xl bg-bg-soft p-7 shadow-card ring-1 ring-navy/5">
                <li className="flex items-start gap-3">
                  <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-navy text-gold">
                    <MapPin className="h-5 w-5" aria-hidden />
                  </span>
                  <div>
                    <p className="font-semibold text-navy">آدرس دفتر</p>
                    <p className="mt-1 text-sm text-muted">{site.address}</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-navy text-gold">
                    <Phone className="h-5 w-5" aria-hidden />
                  </span>
                  <div>
                    <p className="font-semibold text-navy">تلفن</p>
                    <a href={site.phoneHref} dir="ltr" className="mt-1 block text-sm text-muted hover:text-navy">
                      {site.phone}
                    </a>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-navy text-gold">
                    <Mail className="h-5 w-5" aria-hidden />
                  </span>
                  <div>
                    <p className="font-semibold text-navy">ایمیل</p>
                    <a href={`mailto:${site.email}`} dir="ltr" className="mt-1 block text-sm text-muted hover:text-navy">
                      {site.email}
                    </a>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-navy text-gold">
                    <Clock className="h-5 w-5" aria-hidden />
                  </span>
                  <div>
                    <p className="font-semibold text-navy">ساعات کاری</p>
                    <p className="mt-1 text-sm text-muted">{site.hours}</p>
                  </div>
                </li>
              </ul>

              <MapEmbed lat={site.geo.lat} lng={site.geo.lng} title="دفتر افق املاک" />
            </div>
          </Reveal>
        </div>
      </section>
    </>
  );
}
