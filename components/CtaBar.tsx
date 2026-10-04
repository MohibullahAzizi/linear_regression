import Link from "next/link";
import { ArrowLeft, KeyRound } from "lucide-react";

export function CtaBar() {
  return (
    <section className="pb-16 sm:pb-20">
      <div className="container-x">
        <div className="flex flex-col items-center gap-6 rounded-[2rem] bg-bg-soft p-8 text-center shadow-card ring-1 ring-navy/5 sm:p-10 lg:flex-row lg:justify-between lg:text-start">
          <div className="flex flex-col items-center gap-5 lg:flex-row">
            <span className="grid h-16 w-16 shrink-0 place-items-center rounded-full bg-navy text-white">
              <KeyRound className="h-7 w-7" aria-hidden />
            </span>
            <div>
              <h2 className="text-2xl font-extrabold text-navy sm:text-3xl">
                آماده یافتن ملک ایده‌آل خود هستید؟
              </h2>
              <p className="mt-2 text-muted">
                کارشناسان ما شما را به خانه یا سرمایه‌گذاری درست راهنمایی می‌کنند.
              </p>
            </div>
          </div>
          <Link href="/contact" className="btn-navy shrink-0">
            تماس بگیرید
            <ArrowLeft className="h-4 w-4" aria-hidden />
          </Link>
        </div>
      </div>
    </section>
  );
}
