import Link from "next/link";

export default function NotFound() {
  return (
    <section className="flex min-h-[70vh] items-center justify-center bg-bg-soft px-5 pt-24">
      <div className="text-center">
        <p className="text-6xl font-extrabold text-gold">۴۰۴</p>
        <h1 className="mt-4 text-2xl font-extrabold text-navy">صفحه مورد نظر پیدا نشد</h1>
        <p className="mt-3 text-muted">
          ممکن است این ملک حذف شده یا نشانی را اشتباه وارد کرده باشید.
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <Link href="/" className="btn-navy">
            بازگشت به خانه
          </Link>
          <Link href="/properties" className="btn-outline-navy">
            مشاهده املاک
          </Link>
        </div>
      </div>
    </section>
  );
}
