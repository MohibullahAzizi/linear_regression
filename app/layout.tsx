import type { Metadata, Viewport } from "next";
import { Vazirmatn } from "next/font/google";
import "./globals.css";

import { Footer } from "@/components/Footer";
import { JsonLd } from "@/components/JsonLd";
import { Navbar } from "@/components/Navbar";
import { WhatsAppFloat } from "@/components/WhatsAppFloat";
import { site } from "@/lib/site";

const vazirmatn = Vazirmatn({
  subsets: ["arabic", "latin"],
  display: "swap",
  variable: "--font-vazirmatn",
});

export const metadata: Metadata = {
  metadataBase: new URL(site.url),
  title: {
    default: `${site.nameFa} | املاک لوکس و سرمایه‌گذاری مطمئن`,
    template: `%s | ${site.nameFa}`,
  },
  description: site.description,
  keywords: [
    "املاک",
    "خرید خانه",
    "ویلا",
    "آپارتمان لوکس",
    "سرمایه‌گذاری ملکی",
    "مشاور املاک",
    "افق املاک",
  ],
  authors: [{ name: site.name }],
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    locale: "fa_IR",
    url: site.url,
    siteName: site.nameFa,
    title: `${site.nameFa} | املاک لوکس و سرمایه‌گذاری مطمئن`,
    description: site.description,
    images: [
      {
        url: "https://images.unsplash.com/photo-1613490493576-7fde63acd811?auto=format&fit=crop&w=1200&q=80",
        width: 1200,
        height: 630,
        alt: "ویلای لوکس افق املاک",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: `${site.nameFa} | املاک لوکس`,
    description: site.description,
    images: [
      "https://images.unsplash.com/photo-1613490493576-7fde63acd811?auto=format&fit=crop&w=1200&q=80",
    ],
  },
  robots: { index: true, follow: true },
  icons: {
    icon: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%230A2540'/%3E%3Cpath d='M8 22V12l8-5 8 5v10' stroke='%23D4AF37' stroke-width='2.2' fill='none' stroke-linejoin='round'/%3E%3C/svg%3E",
  },
};

export const viewport: Viewport = {
  themeColor: "#0A2540",
  width: "device-width",
  initialScale: 1,
};

const organizationJsonLd = {
  "@context": "https://schema.org",
  "@type": "RealEstateAgent",
  name: site.name,
  description: site.description,
  url: site.url,
  telephone: site.phone,
  email: site.email,
  priceRange: "$$$",
  areaServed: { "@type": "Country", name: "Iran" },
  address: {
    "@type": "PostalAddress",
    streetAddress: site.address,
    addressLocality: "تهران",
    addressCountry: "IR",
  },
  geo: {
    "@type": "GeoCoordinates",
    latitude: site.geo.lat,
    longitude: site.geo.lng,
  },
  openingHours: site.hours,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fa" dir="rtl" className={vazirmatn.variable}>
      <body>
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:start-4 focus:top-4 focus:z-[70] focus:rounded-full focus:bg-navy focus:px-5 focus:py-2 focus:text-sm focus:font-semibold focus:text-white"
        >
          پرش به محتوای اصلی
        </a>
        <Navbar />
        <main id="main">{children}</main>
        <Footer />
        <WhatsAppFloat />
        <JsonLd data={organizationJsonLd} />
      </body>
    </html>
  );
}
