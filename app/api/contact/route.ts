import { NextResponse } from "next/server";
import { deliverLead } from "@/lib/notify";
import { contactSchema } from "@/lib/validation";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const parsed = contactSchema.safeParse(body);

  if (!parsed.success) {
    return NextResponse.json(
      { ok: false, errors: parsed.error.flatten().fieldErrors },
      { status: 400 },
    );
  }

  const d = parsed.data;
  const lines = [
    `نام: ${d.name}`,
    `تلفن: ${d.phone}`,
    `ایمیل: ${d.email || "—"}`,
    `موضوع: ${d.subject || "—"}`,
    "",
    d.message,
  ];

  const delivered = await deliverLead("📩 پیام جدید از فرم تماس", lines);
  console.log("[contact] new lead", JSON.stringify({ ...d, delivered }));

  return NextResponse.json({ ok: true, delivered });
}
