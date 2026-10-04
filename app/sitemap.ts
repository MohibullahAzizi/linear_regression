import type { MetadataRoute } from "next";
import { properties } from "@/lib/properties";
import { site } from "@/lib/site";

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  const routes = ["", "/properties", "/about", "/services", "/team", "/contact", "/favorites"];

  return [
    ...routes.map((route) => ({
      url: `${site.url}${route}`,
      lastModified: now,
      changeFrequency: "weekly" as const,
      priority: route === "" ? 1 : 0.8,
    })),
    ...properties.map((p) => ({
      url: `${site.url}/properties/${p.slug}`,
      lastModified: new Date(p.listedAt),
      changeFrequency: "weekly" as const,
      priority: 0.7,
    })),
  ];
}
