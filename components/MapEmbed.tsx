type Props = {
  lat: number;
  lng: number;
  title: string;
};

/** Key-free OpenStreetMap embed centred on the listing. */
export function MapEmbed({ lat, lng, title }: Props) {
  const d = 0.012;
  const bbox = `${lng - d}%2C${lat - d}%2C${lng + d}%2C${lat + d}`;
  const src = `https://www.openstreetmap.org/export/embed.html?bbox=${bbox}&layer=mapnik&marker=${lat}%2C${lng}`;

  return (
    <div className="overflow-hidden rounded-3xl shadow-card ring-1 ring-navy/5">
      <iframe
        title={`نقشه موقعیت ${title}`}
        src={src}
        loading="lazy"
        className="h-80 w-full border-0"
      />
    </div>
  );
}
