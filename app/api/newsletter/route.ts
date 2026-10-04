import { NextResponse } from "next/server";
import { deliverLead } from "@/lib/notify";
import { newsletterSchema } from "@/lib/validation";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const parsed = newsletterSchema.safeParse(body);

  if (!parsed.success) {
    return NextResponse.json(
      { ok: false, errors: parsed.error.flatten().fieldErrors },
      { status: 400 },
    );
  }

  const delivered = await deliverLead("✉️ عضویت جدید در خبرنامه", [`ایمیل: ${parsed.data.email}`]);
  console.log("[newsletter] new subscriber", JSON.stringify({ email: parsed.data.email, delivered }));

  return NextResponse.json({ ok: true, delivered });
}
