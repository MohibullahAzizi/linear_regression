import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import {
  Bath,
  BedDouble,
  Building2,
  Calendar,
  Car,
  Check,
  ChevronLeft,
  MapPin,
  Maximize,
  Phone,
  User,
} from "lucide-react";

import { FavoriteButton } from "@/components/FavoriteButton";
import { InquiryForm } from "@/components/InquiryForm";
import { JsonLd } from "@/components/JsonLd";
import { MapEmbed } from "@/components/MapEmbed";
import { MortgageCalculator } from "@/components/MortgageCalculator";
import { PropertyCard } from "@/components/PropertyCard";
import { PropertyGallery } from "@/components/PropertyGallery";
import { Reveal } from "@/components/Reveal";
import { SectionHeading } from "@/components/SectionHeading";
import { formatArea, formatJalali, formatToman, toFa } from "@/lib/format";
import { getAllSlugs, getProperty, properties, typeLabel } from "@/lib/properties";
import { site } from "@/lib/site";

type Params = { params: Promise<{ slug: string }> };

export function generateStaticParams() {
  return getAllSlugs().map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { slug } = await params;
  const p = getProperty(slug);
  if (!p) return { title: "ملک پیدا نشد" };
  const description = p.description.slice(0, 155);
  return {
    title: p.title,
    description,
    alternates: { canonical: `/properties/${p.slug}` },
    openGraph: {
      type: "article",
      title: p.title,
      description,
      images: [{ url: p.image, alt: p.title }],
    },
  };
}

export default async function PropertyPage({ params }: Params) {
  const { slug } = await params;
  const property = getProperty(slug);
  if (!property) notFound();
  const p = property;

  const related = properties
    .filter((x) => x.slug !== p.slug && (x.type === p.type || x.city === p.city))
    .slice(0, 3);

  const productJsonLd = {
    "@context": "https://schema.org",
    "@type": "Product",
    name: p.title,
    image: p.gallery,
    description: p.description,
    category: typeLabel(p.type),
    brand: { "@type": "Brand", name: site.name },
    offers: {
      "@type": "Offer",
      price: p.price,
      priceCurrency: "IRR",
      availability: "https://schema.org/InStock",
      url: `${site.url}/properties/${p.slug}`,
      seller: { "@type": "RealEstateAgent", name: site.name },
    },
  };

  const facts = [
    { icon: BedDouble, label: "خواب", value: toFa(p.bedrooms) },
    { icon: Bath, label: "سرویس", value: toFa(p.bathrooms) },
    { icon: Maximize, label: "متراژ", value: formatArea(p.area) },
    { icon: Car, label: "پارکینگ", value: toFa(p.parking) },
    { icon: Calendar, label: "سال ساخت", value: toFa(p.yearBuilt) },
  ];

  return (
    <>
      <article className="bg-white pt-24 lg:pt-32">
        <div className="container-x">
          <nav aria-label="مسیر صفحه" className="flex items-center gap-1 text-xs text-muted">
            <Link href="/" className="hover:text-navy">خانه</Link>
            <ChevronLeft className="h-3.5 w-3.5" aria-hidden />
            <Link href="/properties" className="hover:text-navy">املاک</Link>
            <ChevronLeft className="h-3.5 w-3.5" aria-hidden />
            <span className="text-navy">{p.title}</span>
          </nav>

          <div className="mt-6">
            <PropertyGallery images={p.gallery} title={p.title} />
          </div>

          <div className="mt-10 grid gap-10 lg:grid-cols-3">
            {/* Main */}
            <div className="lg:col-span-2">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-navy/5 px-3 py-1 text-xs font-semibold text-navy">
                    <Building2 className="h-3.5 w-3.5" aria-hidden />
                    {typeLabel(p.type)}
                  </span>
                  <h1 className="mt-3 text-2xl font-extrabold text-navy sm:text-3xl">{p.title}</h1>
                  <p className="mt-2 flex items-center gap-1.5 text-muted">
                    <MapPin className="h-4 w-4 text-gold" aria-hidden />
                    {p.location}
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  <p className="text-xl font-extrabold text-navy sm:text-2xl">{formatToman(p.price)}</p>
                  <FavoriteButton slug={p.slug} title={p.title} />
                </div>
              </div>

              <dl className="mt-8 grid grid-cols-2 gap-4 rounded-3xl bg-bg-soft p-5 sm:grid-cols-5">
                {facts.map((f) => (
                  <div key={f.label} className="flex flex-col items-center gap-1 text-center">
                    <f.icon className="h-5 w-5 text-gold" aria-hidden />
                    <dt className="text-xs text-muted">{f.label}</dt>
                    <dd className="text-sm font-bold text-navy">{f.value}</dd>
                  </div>
                ))}
              </dl>

              <div className="mt-10">
                <h2 className="text-xl font-extrabold text-navy">توضیحات</h2>
                <p className="mt-4 leading-8 text-muted">{p.description}</p>
                <p className="mt-3 text-sm text-muted">
                  تاریخ انتشار: {formatJalali(p.listedAt)}
                </p>
              </div>

              <div className="mt-10">
                <h2 className="text-xl font-extrabold text-navy">امکانات</h2>
                <ul className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {p.features.map((f) => (
                    <li key={f} className="flex items-center gap-2 text-sm text-text">
                      <span className="grid h-6 w-6 place-items-center rounded-full bg-gold/15 text-gold">
                        <Check className="h-3.5 w-3.5" aria-hidden />
                      </span>
                      {f}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="mt-10">
                <h2 className="mb-4 text-xl font-extrabold text-navy">موقعیت روی نقشه</h2>
                <MapEmbed lat={p.geo.lat} lng={p.geo.lng} title={p.title} />
              </div>
            </div>

            {/* Sidebar */}
            <aside className="space-y-6 lg:sticky lg:top-28 lg:self-start">
              <div className="rounded-3xl bg-navy p-6 text-white shadow-card">
                <div className="flex items-center gap-3">
                  <span className="grid h-12 w-12 place-items-center rounded-full bg-white/10 text-gold">
                    <User className="h-6 w-6" aria-hidden />
                  </span>
                  <div>
                    <p className="font-bold">{p.agent.name}</p>
                    <p className="text-xs text-white/70">{p.agent.title}</p>
                  </div>
                </div>
                <a
                  href={site.phoneHref}
                  className="btn mt-5 w-full bg-gold text-navy hover:bg-white"
                >
                  <Phone className="h-4 w-4" aria-hidden />
                  <span dir="ltr">{site.phone}</span>
                </a>
              </div>

              <div className="rounded-3xl bg-white p-6 shadow-card ring-1 ring-navy/5">
                <h2 className="text-lg font-extrabold text-navy">درخواست بازدید</h2>
                <p className="mt-1 text-sm text-muted">فرم را پر کنید تا با شما تماس بگیریم.</p>
                <div className="mt-5">
                  <InquiryForm propertySlug={p.slug} propertyTitle={p.title} />
                </div>
              </div>

              <MortgageCalculator price={p.price} />
            </aside>
          </div>
        </div>
      </article>

      {related.length > 0 && (
        <section className="section bg-bg-soft">
          <div className="container-x">
            <Reveal>
              <SectionHeading eyebrow="پیشنهاد ما" title="املاک مشابه" />
            </Reveal>
            <div className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
              {related.map((r) => (
                <PropertyCard key={r.slug} property={r} />
              ))}
            </div>
          </div>
        </section>
      )}

      <JsonLd data={productJsonLd} />
    </>
  );
}
