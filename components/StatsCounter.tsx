import { AnimatedCounter } from "@/components/AnimatedCounter";

const stats = [
  { value: 1250, suffix: "+", label: "ملک فروخته‌شده" },
  { value: 18, suffix: "", label: "سال تجربه" },
  { value: 96, suffix: "٪", label: "رضایت مشتریان" },
  { value: 42, suffix: "", label: "کارشناس متخصص" },
];

export function StatsCounter() {
  return (
    <section className="bg-navy py-14" aria-label="آمار افق املاک">
      <div className="container-x grid grid-cols-2 gap-8 lg:grid-cols-4">
        {stats.map((s) => (
          <div key={s.label} className="text-center text-white">
            <p className="text-3xl font-extrabold text-gold sm:text-4xl">
              <AnimatedCounter value={s.value} suffix={s.suffix} />
            </p>
            <p className="mt-2 text-sm text-white/70">{s.label}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
