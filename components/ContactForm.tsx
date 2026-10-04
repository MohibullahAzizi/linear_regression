"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { CheckCircle2, Loader2, Send } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { errorClass, inputClass, labelClass } from "@/components/formStyles";
import { contactSchema, type ContactInput } from "@/lib/validation";

export function ContactForm() {
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<ContactInput>({ resolver: zodResolver(contactSchema) });

  const [status, setStatus] = useState<"idle" | "success" | "error">("idle");

  const onSubmit = async (data: ContactInput) => {
    setStatus("idle");
    try {
      const res = await fetch("/api/contact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("request failed");
      setStatus("success");
      reset();
    } catch {
      setStatus("error");
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} noValidate className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <label htmlFor="c-name" className={labelClass}>
            نام و نام خانوادگی
          </label>
          <input id="c-name" className={inputClass} placeholder="مثلاً سارا محمدی" {...register("name")} />
          {errors.name && <p className={errorClass}>{errors.name.message}</p>}
        </div>
        <div>
          <label htmlFor="c-phone" className={labelClass}>
            شماره تماس
          </label>
          <input
            id="c-phone"
            dir="ltr"
            inputMode="tel"
            className={`${inputClass} text-start`}
            placeholder="09123456789"
            {...register("phone")}
          />
          {errors.phone && <p className={errorClass}>{errors.phone.message}</p>}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <label htmlFor="c-email" className={labelClass}>
            ایمیل (اختیاری)
          </label>
          <input id="c-email" dir="ltr" type="email" className={`${inputClass} text-start`} placeholder="you@example.com" {...register("email")} />
          {errors.email && <p className={errorClass}>{errors.email.message}</p>}
        </div>
        <div>
          <label htmlFor="c-subject" className={labelClass}>
            موضوع (اختیاری)
          </label>
          <input id="c-subject" className={inputClass} placeholder="مشاوره خرید ویلا" {...register("subject")} />
        </div>
      </div>

      <div>
        <label htmlFor="c-message" className={labelClass}>
          پیام شما
        </label>
        <textarea id="c-message" rows={5} className={inputClass} placeholder="چطور می‌توانیم کمک کنیم؟" {...register("message")} />
        {errors.message && <p className={errorClass}>{errors.message.message}</p>}
      </div>

      <button type="submit" disabled={isSubmitting} className="btn-navy w-full sm:w-auto">
        {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <Send className="h-4 w-4" aria-hidden />}
        ارسال پیام
      </button>

      <div aria-live="polite">
        {status === "success" && (
          <p className="flex items-center gap-2 text-sm font-medium text-green-700">
            <CheckCircle2 className="h-4 w-4" aria-hidden />
            پیام شما ارسال شد. به‌زودی با شما تماس می‌گیریم.
          </p>
        )}
        {status === "error" && (
          <p className={errorClass}>ارسال پیام ناموفق بود. لطفاً دوباره تلاش کنید.</p>
        )}
      </div>
    </form>
  );
}
