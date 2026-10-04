# Base44 Dev Environment

This repo hosts a **Next.js 15 Persian RTL real-estate website** ("Horizon
Properties" / افق املاک) served on host port 3000, alongside the original
academic AI/ML Python notebooks and scripts.

## Running

```
docker compose -f docker-compose.base44.yml up -d --build
```

- **Web (Next.js dev):** `http://localhost:3000` — `node:22` base image, repo
  bind-mounted at `/app`, `npm install` then `npm run dev` (next dev). Live
  reload via WATCHPACK_POLLING.
- **Jupyter (optional):** `http://localhost:8888` — Python 3.12 + JupyterLab
  for the academic notebooks. Not required for the web app.

## What was fixed

The `lib/` directory was never committed when the real-estate site was built,
causing a 500 error (`Module not found: Can't resolve '@/lib/format'`). The
following modules were recreated from usage in `app/` and `components/`:

- `lib/types.ts` — `Property`, `PropertyType` types
- `lib/cn.ts` — `cn` className helper (clsx)
- `lib/format.ts` — Persian digit conversion, Toman/Jalali/area formatting
- `lib/site.ts` — site config + nav links
- `lib/properties.ts` — property data loading, filtering, sorting
- `lib/favorites.ts` — localStorage favorites management
- `lib/validation.ts` — Zod schemas for contact/inquiry/newsletter forms
- `lib/notify.ts` — lead delivery (Telegram / CRM webhook)

Next.js was also upgraded from 15.1.6 → 15.5.4 so `allowedDevOrigins` in
`next.config.ts` is recognized (required for the Base44 preview origin).

## Verifying it works

```
docker compose -f docker-compose.base44.yml ps
curl -sS -o /dev/null -w "%{http_code}" http://localhost:3000/
```
Should return `200`. Check all routes: `/`, `/properties`, `/about`,
`/services`, `/team`, `/contact`, `/favorites`.

## Notes

- Optional secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `CRM_WEBHOOK_URL`
  for lead delivery. The site runs without them.
- Property data is in `data/properties.json` (12 listings).
- Images use Unsplash remote URLs (configured in `next.config.ts`).
