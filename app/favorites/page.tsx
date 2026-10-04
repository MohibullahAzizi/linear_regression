import type { Metadata } from "next";
import { FavoritesList } from "@/components/FavoritesList";
import { PageHero } from "@/components/PageHero";
import { properties } from "@/lib/properties";

export const metadata: Metadata = {
  title: "علاقه‌مندی‌ها",
  description: "املاکی که ذخیره کرده‌اید.",
  alternates: { canonical: "/favorites" },
  robots: { index: false, follow: true },
};

export default function FavoritesPage() {
  return (
    <>
      <PageHero
        eyebrow="ذخیره‌شده‌ها"
        title="علاقه‌مندی‌های من"
        description="املاکی که با زدن آیکون قلب ذخیره کرده‌اید، در همین مرورگر نگهداری می‌شوند."
      />
      <section className="section bg-white">
        <div className="container-x">
          <FavoritesList all={properties} />
        </div>
      </section>
    </>
  );
}
