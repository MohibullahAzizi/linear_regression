"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { CheckCircle2, Loader2, Send } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { errorClass, inputClass, labelClass } from "@/components/formStyles";
import { inquirySchema, type InquiryInput } from "@/lib/validation";

type Props = {
  propertySlug: string;
  propertyTitle: string;
};

export function InquiryForm({ propertySlug, propertyTitle }: Props) {
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<InquiryInput>({
    resolver: zodResolver(inquirySchema),
    defaultValues: { propertySlug, propertyTitle, message: `سلام، درباره «${propertyTitle}» اطلاعات بیشتری می‌خواهم.` },
  });

  const [status, setStatus] = useState<"idle" | "success" | "error">("idle");

  const onSubmit = async (data: InquiryInput) => {
    setStatus("idle");
    try {
      const res = await fetch("/api/inquiry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("request failed");
      setStatus("success");
      reset({ propertySlug, propertyTitle });
    } catch {
      setStatus("error");
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} noValidate className="space-y-4">
      <input type="hidden" {...register("propertySlug")} />
      <input type="hidden" {...register("propertyTitle")} />

      <div>
        <label htmlFor="i-name" className={labelClass}>
          نام و نام خانوادگی
        </label>
        <input id="i-name" className={inputClass} placeholder="مثلاً امیر رضایی" {...register("name")} />
        {errors.name && <p className={errorClass}>{errors.name.message}</p>}
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <label htmlFor="i-phone" className={labelClass}>
            شماره تماس
          </label>
          <input id="i-phone" dir="ltr" inputMode="tel" className={`${inputClass} text-start`} placeholder="09123456789" {...register("phone")} />
          {errors.phone && <p className={errorClass}>{errors.phone.message}</p>}
        </div>
        <div>
          <label htmlFor="i-email" className={labelClass}>
            ایمیل (اختیاری)
          </label>
          <input id="i-email" dir="ltr" type="email" className={`${inputClass} text-start`} placeholder="you@example.com" {...register("email")} />
          {errors.email && <p className={errorClass}>{errors.email.message}</p>}
        </div>
      </div>

      <div>
        <label htmlFor="i-message" className={labelClass}>
          پیام
        </label>
        <textarea id="i-message" rows={4} className={inputClass} {...register("message")} />
        {errors.message && <p className={errorClass}>{errors.message.message}</p>}
      </div>

      <button type="submit" disabled={isSubmitting} className="btn-navy w-full">
        {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <Send className="h-4 w-4" aria-hidden />}
        درخواست بازدید
      </button>

      <div aria-live="polite">
        {status === "success" && (
          <p className="flex items-center gap-2 text-sm font-medium text-green-700">
            <CheckCircle2 className="h-4 w-4" aria-hidden />
            درخواست شما ثبت شد. کارشناس ما با شما تماس می‌گیرد.
          </p>
        )}
        {status === "error" && <p className={errorClass}>ثبت درخواست ناموفق بود. دوباره تلاش کنید.</p>}
      </div>
    </form>
  );
}
