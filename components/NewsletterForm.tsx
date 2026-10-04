"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { CheckCircle2, Loader2, Send } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { errorClass, inputClass } from "@/components/formStyles";
import { newsletterSchema, type NewsletterInput } from "@/lib/validation";

export function NewsletterForm() {
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<NewsletterInput>({ resolver: zodResolver(newsletterSchema) });
  const [status, setStatus] = useState<"idle" | "success" | "error">("idle");

  const onSubmit = async (data: NewsletterInput) => {
    setStatus("idle");
    try {
      const res = await fetch("/api/newsletter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error();
      setStatus("success");
      reset();
    } catch {
      setStatus("error");
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} noValidate className="mt-4">
      <label htmlFor="n-email" className="sr-only">
        ایمیل برای خبرنامه
      </label>
      <div className="flex gap-2">
        <input
          id="n-email"
          dir="ltr"
          type="email"
          className={`${inputClass} text-start`}
          placeholder="you@example.com"
          {...register("email")}
        />
        <button
          type="submit"
          disabled={isSubmitting}
          aria-label="عضویت در خبرنامه"
          className="btn shrink-0 bg-gold text-navy hover:bg-white"
        >
          {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <Send className="h-4 w-4" aria-hidden />}
        </button>
      </div>
      <div aria-live="polite">
        {errors.email && <p className={errorClass}>{errors.email.message}</p>}
        {status === "success" && (
          <p className="mt-2 flex items-center gap-2 text-xs font-medium text-green-400">
            <CheckCircle2 className="h-3.5 w-3.5" aria-hidden />
            عضویت شما ثبت شد.
          </p>
        )}
        {status === "error" && <p className={errorClass}>ثبت‌نام ناموفق بود.</p>}
      </div>
    </form>
  );
}
