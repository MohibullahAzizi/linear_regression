import Image from "next/image";
import Link from "next/link";
import { Bath, BedDouble, Car, MapPin, Maximize } from "lucide-react";
import { FavoriteButton } from "@/components/FavoriteButton";
import { formatArea, formatToman, toFa } from "@/lib/format";
import { typeLabel } from "@/lib/properties";
import type { Property } from "@/lib/types";

type Props = {
  property: Property;
  priority?: boolean;
};

export function PropertyCard({ property: p, priority }: Props) {
  return (
    <article className="group relative h-full overflow-hidden rounded-3xl bg-white shadow-card ring-1 ring-navy/5 transition-all duration-500 hover:-translate-y-1 hover:scale-[1.02] hover:shadow-card-hover">
      <Link href={`/properties/${p.slug}`} className="block h-full">
        <div className="relative aspect-[4/3] overflow-hidden">
          <Image
            src={p.image}
            alt={`${p.title} — ${p.location}`}
            fill
            sizes="(max-width:640px) 100vw, (max-width:1024px) 50vw, 33vw"
            priority={priority}
            className="object-cover transition-transform duration-700 group-hover:scale-110"
          />
          <span className="absolute start-4 top-4 rounded-full bg-navy/85 px-3 py-1 text-xs font-semibold text-white backdrop-blur">
            {typeLabel(p.type)}
          </span>
        </div>

        <div className="p-5">
          <p className="text-lg font-extrabold text-navy">{formatToman(p.price)}</p>
          <h3 className="mt-1 line-clamp-1 font-bold text-text">{p.title}</h3>
          <p className="mt-1 flex items-center gap-1 text-sm text-muted">
            <MapPin className="h-4 w-4 shrink-0 text-gold" aria-hidden />
            {p.location}
          </p>

          <ul className="mt-4 flex flex-wrap items-center justify-between gap-y-2 border-t border-navy/5 pt-4 text-xs text-muted">
            <li className="flex items-center gap-1">
              <BedDouble className="h-4 w-4" aria-hidden />
              {toFa(p.bedrooms)} خواب
            </li>
            <li className="flex items-center gap-1">
              <Bath className="h-4 w-4" aria-hidden />
              {toFa(p.bathrooms)} سرویس
            </li>
            <li className="flex items-center gap-1">
              <Maximize className="h-4 w-4" aria-hidden />
              {formatArea(p.area)}
            </li>
            <li className="flex items-center gap-1">
              <Car className="h-4 w-4" aria-hidden />
              {toFa(p.parking)} پارکینگ
            </li>
          </ul>
        </div>
      </Link>

      <div className="absolute end-4 top-4">
        <FavoriteButton slug={p.slug} title={p.title} />
      </div>
    </article>
  );
}
