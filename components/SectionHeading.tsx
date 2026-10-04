import { cn } from "@/lib/cn";

type Props = {
  eyebrow?: string;
  title: string;
  description?: string;
  align?: "center" | "start";
  className?: string;
  as?: "h2" | "h1";
};

export function SectionHeading({
  eyebrow,
  title,
  description,
  align = "center",
  className,
  as: Tag = "h2",
}: Props) {
  return (
    <div
      className={cn(
        "max-w-2xl",
        align === "center" ? "mx-auto text-center" : "text-start",
        className,
      )}
    >
      {eyebrow && <p className="eyebrow mb-3">{eyebrow}</p>}
      <Tag className="text-3xl font-extrabold leading-tight text-navy sm:text-4xl">{title}</Tag>
      {description && <p className="mt-4 leading-7 text-muted">{description}</p>}
    </div>
  );
}
