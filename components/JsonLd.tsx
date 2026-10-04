type Props = {
  data: Record<string, unknown> | Record<string, unknown>[];
};

/** Injects a JSON-LD structured-data script. */
export function JsonLd({ data }: Props) {
  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(data) }}
    />
  );
}
