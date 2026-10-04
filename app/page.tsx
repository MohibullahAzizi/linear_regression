import { About } from "@/components/About";
import { CtaBar } from "@/components/CtaBar";
import { FeaturedCarousel } from "@/components/FeaturedCarousel";
import { Hero } from "@/components/Hero";
import { Reveal } from "@/components/Reveal";
import { SectionHeading } from "@/components/SectionHeading";
import { StatsCounter } from "@/components/StatsCounter";
import { getFeatured } from "@/lib/properties";

export default function HomePage() {
  const featured = getFeatured(8);

  return (
    <>
      <Hero />
      <About />

      <section className="section bg-bg-soft" id="featured">
        <div className="container-x">
          <Reveal>
            <SectionHeading
              eyebrow="ویژه"
              title="املاک ویژه"
              description="منتخبی از بهترین خانه‌ها و فرصت‌های سرمایه‌گذاری، دست‌چین‌شده توسط کارشناسان افق املاک."
            />
          </Reveal>
          <div className="mt-12">
            <FeaturedCarousel items={featured} />
          </div>
        </div>
      </section>

      <StatsCounter />
      <CtaBar />
    </>
  );
}
