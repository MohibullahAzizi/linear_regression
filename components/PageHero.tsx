type Props = {
  eyebrow?: string;
  title: string;
  description?: string;
};

/** Navy page banner for inner pages; its top padding clears the fixed navbar. */
export function PageHero({ eyebrow, title, description }: Props) {
  return (
    <section className="bg-navy pb-14 pt-28 text-white lg:pt-36">
      <div className="container-x text-center">
        {eyebrow && <p className="eyebrow mb-3">{eyebrow}</p>}
        <h1 className="mx-auto max-w-3xl text-3xl font-extrabold leading-tight sm:text-4xl lg:text-5xl">
          {title}
        </h1>
        {description && (
          <p className="mx-auto mt-4 max-w-2xl leading-8 text-white/80">{description}</p>
        )}
      </div>
    </section>
  );
}
