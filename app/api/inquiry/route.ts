import { NextResponse } from "next/server";
import { deliverLead } from "@/lib/notify";
import { inquirySchema } from "@/lib/validation";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const parsed = inquirySchema.safeParse(body);

  if (!parsed.success) {
    return NextResponse.json(
      { ok: false, errors: parsed.error.flatten().fieldErrors },
      { status: 400 },
    );
  }

  const d = parsed.data;
  const lines = [
    `ملک: ${d.propertyTitle} (${d.propertySlug})`,
    `نام: ${d.name}`,
    `تلفن: ${d.phone}`,
    `ایمیل: ${d.email || "—"}`,
    "",
    d.message,
  ];

  const delivered = await deliverLead("🏠 درخواست بازدید ملک", lines);
  console.log("[inquiry] new lead", JSON.stringify({ ...d, delivered }));

  return NextResponse.json({ ok: true, delivered });
}
