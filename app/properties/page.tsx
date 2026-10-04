import type { Metadata } from "next";
import { Suspense } from "react";
import { PageHero } from "@/components/PageHero";
import { PropertiesExplorer } from "@/components/PropertiesExplorer";
import { properties } from "@/lib/properties";

export const metadata: Metadata = {
  title: "املاک",
  description:
    "همه املاک افق املاک: ویلا، آپارتمان، پنت‌هاوس و دفاتر اداری در بهترین موقعیت‌های شهری. جست‌وجو، فیلتر و مرتب‌سازی کنید.",
  alternates: { canonical: "/properties" },
};

export default function PropertiesPage() {
  return (
    <>
      <PageHero
        eyebrow="املاک"
        title="همه املاک"
        description="با فیلترهای هوشمند، ملک مناسب خود را پیدا کنید؛ از ویلاهای لوکس تا آپارتمان‌های خانوادگی."
      />
      <section className="section bg-white">
        <div className="container-x">
          <Suspense fallback={<p className="py-16 text-center text-muted">در حال بارگذاری املاک…</p>}>
            <PropertiesExplorer all={properties} />
          </Suspense>
        </div>
      </section>
    </>
  );
}
