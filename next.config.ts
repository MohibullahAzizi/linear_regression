import type { NextConfig } from "next";

const hostSuffix = process.env.BASE44_PUBLIC_HOST_SUFFIX;

const nextConfig: NextConfig = {
  images: {
    formats: ["image/webp"],
    remotePatterns: [
      { protocol: "https", hostname: "images.unsplash.com" },
    ],
  },
  // Allow the Base44 preview origin (https://3000-<suffix>) to load dev assets / HMR.
  allowedDevOrigins: hostSuffix ? [`3000-${hostSuffix}`] : [],
};

export default nextConfig;
