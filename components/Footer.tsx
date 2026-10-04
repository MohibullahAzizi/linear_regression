import Link from "next/link";
import { Building2, Mail, MapPin, Phone } from "lucide-react";
import { NewsletterForm } from "@/components/NewsletterForm";
import { jalaliYear } from "@/lib/format";
import { navLinks, site } from "@/lib/site";

export function Footer() {
  return (
    <footer className="bg-navy-dark text-white/75">
      <div className="container-x grid gap-10 py-14 md:grid-cols-2 lg:grid-cols-4">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="grid h-10 w-10 place-items-center rounded-xl bg-white/10 text-gold">
              <Building2 className="h-5 w-5" aria-hidden />
            </span>
            <span className="text-sm font-extrabold tracking-[0.18em] text-white">
              HORIZON PROPERTIES
            </span>
          </div>
          <p className="mt-4 text-sm leading-7">
            {site.description}
          </p>
        </div>

        <nav aria-label="پیوندهای پانوشت">
          <h2 className="text-sm font-bold text-white">دسترسی سریع</h2>
          <ul className="mt-4 space-y-2 text-sm">
            {navLinks.map((l) => (
              <li key={l.href}>
                <Link href={l.href} className="transition hover:text-gold">
                  {l.label}
                </Link>
              </li>
            ))}
            <li>
              <Link href="/favorites" className="transition hover:text-gold">
                علاقه‌مندی‌ها
              </Link>
            </li>
          </ul>
        </nav>

        <div>
          <h2 className="text-sm font-bold text-white">تماس با ما</h2>
          <ul className="mt-4 space-y-3 text-sm">
            <li className="flex items-start gap-2">
              <MapPin className="mt-0.5 h-4 w-4 shrink-0 text-gold" aria-hidden />
              {site.address}
            </li>
            <li>
              <a href={site.phoneHref} className="flex items-center gap-2 transition hover:text-gold">
                <Phone className="h-4 w-4 text-gold" aria-hidden />
                <span dir="ltr">{site.phone}</span>
              </a>
            </li>
            <li>
              <a href={`mailto:${site.email}`} className="flex items-center gap-2 transition hover:text-gold">
                <Mail className="h-4 w-4 text-gold" aria-hidden />
                <span dir="ltr">{site.email}</span>
              </a>
            </li>
          </ul>
        </div>

        <div>
          <h2 className="text-sm font-bold text-white">خبرنامه</h2>
          <p className="mt-4 text-sm">جدیدترین املاک و فرصت‌های سرمایه‌گذاری را دریافت کنید.</p>
          <NewsletterForm />
        </div>
      </div>

      <div className="border-t border-white/10">
        <div className="container-x flex flex-col items-center justify-between gap-2 py-6 text-center text-xs sm:flex-row sm:text-start">
          <p>
            © {jalaliYear()} {site.name} — تمامی حقوق محفوظ است.
          </p>
          <p>ساخته‌شده با ❤️ برای بازار املاک ایران</p>
        </div>
      </div>
    </footer>
  );
}
