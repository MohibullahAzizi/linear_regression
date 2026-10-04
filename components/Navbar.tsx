"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Building2, Heart, Menu, Phone, X } from "lucide-react";
import { cn } from "@/lib/cn";
import { FAVORITES_EVENT, getFavorites } from "@/lib/favorites";
import { navLinks, site } from "@/lib/site";

export function Navbar() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const [favCount, setFavCount] = useState(0);

  const isHome = pathname === "/";
  const solid = scrolled || !isHome || open;

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const sync = () => setFavCount(getFavorites().length);
    sync();
    window.addEventListener(FAVORITES_EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(FAVORITES_EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  useEffect(() => setOpen(false), [pathname]);

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href);

  return (
    <header
      className={cn(
        "fixed inset-x-0 top-0 z-50 transition-all duration-300",
        solid
          ? "bg-white/95 shadow-[0_10px_30px_-20px_rgba(10,37,64,0.5)] backdrop-blur"
          : "bg-transparent",
      )}
    >
      <nav className="container-x flex h-16 items-center justify-between gap-4 lg:h-20" aria-label="ناوبری اصلی">
        {/* Logo — appears on the right in RTL */}
        <Link href="/" className="flex items-center gap-2.5" aria-label={`${site.name} — خانه`}>
          <span
            className={cn(
              "grid h-10 w-10 place-items-center rounded-xl transition-colors",
              solid ? "bg-navy text-gold" : "bg-white/15 text-gold backdrop-blur",
            )}
          >
            <Building2 className="h-5 w-5" aria-hidden />
          </span>
          <span
            className={cn(
              "text-sm font-extrabold tracking-[0.18em] transition-colors sm:text-base",
              solid ? "text-navy" : "text-white",
            )}
          >
            HORIZON PROPERTIES
          </span>
        </Link>

        {/* Desktop nav */}
        <ul className="hidden items-center gap-1 lg:flex">
          {navLinks.map((link) => (
            <li key={link.href}>
              <Link
                href={link.href}
                aria-current={isActive(link.href) ? "page" : undefined}
                className={cn(
                  "relative rounded-full px-4 py-2 text-sm font-semibold transition-colors",
                  solid ? "text-navy/80 hover:text-navy" : "text-white/85 hover:text-white",
                )}
              >
                {link.label}
                <span
                  className={cn(
                    "absolute inset-x-4 -bottom-0.5 h-0.5 rounded-full bg-gold transition-transform duration-300",
                    isActive(link.href) ? "scale-x-100" : "scale-x-0",
                  )}
                />
              </Link>
            </li>
          ))}
        </ul>

        {/* Actions — far side (left in RTL) */}
        <div className="flex items-center gap-2">
          <Link
            href="/favorites"
            aria-label={`علاقه‌مندی‌ها (${favCount})`}
            className={cn(
              "relative hidden h-10 w-10 place-items-center rounded-full transition sm:grid",
              solid ? "text-navy hover:bg-navy/5" : "text-white hover:bg-white/15",
            )}
          >
            <Heart className="h-5 w-5" aria-hidden />
            {favCount > 0 && (
              <span className="absolute -end-0.5 -top-0.5 grid h-5 min-w-5 place-items-center rounded-full bg-gold px-1 text-[10px] font-bold text-navy">
                {favCount.toLocaleString("fa-IR")}
              </span>
            )}
          </Link>

          <a
            href={site.phoneHref}
            className={cn(
              "btn hidden !px-4 !py-2.5 text-xs md:inline-flex",
              solid ? "btn-gold" : "btn-gold",
            )}
          >
            <Phone className="h-4 w-4" aria-hidden />
            <span dir="ltr">{site.phone}</span>
          </a>

          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            aria-controls="mobile-drawer"
            aria-label={open ? "بستن منو" : "باز کردن منو"}
            className={cn(
              "grid h-10 w-10 place-items-center rounded-full transition lg:hidden",
              solid ? "text-navy hover:bg-navy/5" : "text-white hover:bg-white/15",
            )}
          >
            {open ? <X className="h-6 w-6" aria-hidden /> : <Menu className="h-6 w-6" aria-hidden />}
          </button>
        </div>
      </nav>

      {/* Mobile drawer */}
      <AnimatePresence>
        {open && (
          <motion.div
            id="mobile-drawer"
            initial={{ opacity: 0, x: "100%" }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: "100%" }}
            transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
            className="fixed inset-y-0 right-0 z-50 w-[82%] max-w-sm overflow-y-auto bg-white p-6 shadow-2xl lg:hidden"
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-extrabold tracking-[0.18em] text-navy">HORIZON</span>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label="بستن منو"
                className="grid h-10 w-10 place-items-center rounded-full text-navy hover:bg-navy/5"
              >
                <X className="h-6 w-6" aria-hidden />
              </button>
            </div>
            <ul className="mt-8 space-y-1">
              {navLinks.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className={cn(
                      "block rounded-xl px-4 py-3 text-base font-semibold transition-colors",
                      isActive(link.href)
                        ? "bg-navy/5 text-navy"
                        : "text-text hover:bg-bg-soft",
                    )}
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
              <li>
                <Link
                  href="/favorites"
                  className="flex items-center gap-2 rounded-xl px-4 py-3 text-base font-semibold text-text hover:bg-bg-soft"
                >
                  <Heart className="h-5 w-5 text-gold" aria-hidden />
                  علاقه‌مندی‌ها ({favCount.toLocaleString("fa-IR")})
                </Link>
              </li>
            </ul>
            <a href={site.phoneHref} className="btn-navy mt-6 w-full">
              <Phone className="h-4 w-4" aria-hidden />
              <span dir="ltr">{site.phone}</span>
            </a>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
