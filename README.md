# افق املاک — وب‌سایت املاک لوکس (HORIZON PROPERTIES)

وب‌سایت کاملاً فارسی و راست‌به‌چپ (RTL) برای یک آژانس املاک لوکس، ساخته‌شده با
**Next.js (App Router) + TypeScript + Tailwind CSS**.

## راه‌اندازی

```bash
npm install
npm run dev      # http://localhost:3000
npm run build && npm start
```

در محیط Base44، اپ با Docker Compose اجرا می‌شود:

```bash
docker compose -f docker-compose.base44.yml up -d --build
```

- سایت (Next.js): پورت `3000`
- نوت‌بوک‌ها (JupyterLab): پورت `8888`

## ساختار پروژه

```
app/                 # مسیرها (App Router)
  page.tsx           # صفحه اصلی (هیرو، درباره ما، املاک ویژه، آمار، CTA)
  properties/        # فهرست املاک + صفحه جزئیات /properties/[slug]
  about|services|team|contact|favorites/
  api/               # contact | inquiry | newsletter  (اعتبارسنجی + ارسال سرنخ)
  sitemap.ts robots.ts
components/          # Navbar، Hero، SectionHeading، PropertyCard، Carousel،
                     # StatsCounter، ContactForm، Footer، MortgageCalculator، ...
data/properties.json # ۱۲ ملک نمونه
lib/                 # site، properties، format (اعداد فارسی/تقویم جلالی)، validation، ...
```

## امکانات

- **RTL کامل + فونت وزیرمتن**، اعداد فارسی، قیمت به تومان، تاریخ جلالی.
- **ناوبری**: شفاف روی هیرو، تیره پس از اسکرول، خط طلایی لینک فعال، منوی کشویی موبایل.
- **کاروسل املاک ویژه** (Swiper) با فلش‌های معکوس، درگ/سوایپ و اسلاید بعدی نمایان.
- **فیلتر/مرتب‌سازی/صفحه‌بندی سمت کلاینت** در `/properties` با پارامترهای URL.
- **صفحه جزئیات**: گالری با لایت‌باکس، نقشه، ماشین‌حساب اقساط، فرم درخواست بازدید.
- **فرم‌ها** با react-hook-form + zod و API Routes آماده ارسال به **تلگرام** یا **CRM**.
- **علاقه‌مندی‌ها** در localStorage، دکمه شناور واتساپ، لینک تماس مستقیم.
- **SEO**: متا/Open Graph، sitemap.xml، robots.txt، داده ساخت‌یافته JSON-LD.
- **دسترس‌پذیری** و انیمیشن‌های ظریف (fade-up، شمارنده‌های متحرک).

## متغیرهای محیطی (اختیاری)

برای فعال‌کردن ارسال سرنخ‌ها، این‌ها را تنظیم کنید (در غیر این صورت فرم‌ها فقط لاگ می‌شوند):

| متغیر | توضیح |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | توکن ربات تلگرام (از @BotFather) |
| `TELEGRAM_CHAT_ID` | شناسه چت/گروه برای دریافت پیام‌ها |
| `CRM_WEBHOOK_URL` | آدرس وب‌هوک برای ارسال JSON سرنخ |

---

# Fundamental of Artificial Intelligence - Phase 1 Projects

This repository contains three distinct AI projects covering fundamental concepts in search algorithms, machine learning, and optimization techniques.

## Project Overview

### 1. **Angry Birds: Star Wars - Search Algorithms**
**Objective**: Help Luke Skywalker navigate through a grid-based environment to collect all eggs while avoiding obstacles and enemies.

**Key Components**:
- Grid-based environment with various obstacles (bushes, boxes, moving pigs)
- Multiple search algorithms to implement:
  - **BFS** (Breadth-First Search)
  - **UCS** (Uniform Cost Search) 
  - **DLS** (Depth-Limited Search)
  - **A*** (with custom heuristic)
- Environment interaction through provided helper functions
- Evaluation based on expanded nodes and path efficiency

**Technologies**: PyGame, NumPy

### 2. **Linear Regression - Asteroid Diameter Prediction**
**Objective**: Develop a linear regression model to predict asteroid diameters using orbital and physical characteristics.

**Key Components**:
- Dataset containing asteroid features (orbital parameters, absolute magnitude, albedo, etc.)
- Implementation of **Stochastic Gradient Descent (SGD)** from scratch
- Data preprocessing and exploratory data analysis
- Model evaluation using R², MAE, and MSE metrics
- Optional advanced features: momentum, learning rate scheduling, early stopping, regularization

### 3. **DNA Center Finding - Local Search Algorithms** 
**Objective**: Find the central DNA string that minimizes the maximum Hamming distance to all strings in a set using local search algorithms.

**Key Components**:
- Implementation of two local search algorithms:
  - **Hill Climbing** (greedy local search)
  - **Simulated Annealing** (probabilistic optimization)
- Problem formulation based on Closest String Problem
- Functions for neighbor generation and cost calculation
- Comparison with brute-force approach
- Convergence analysis and parameter tuning

## Learning Objectives

- **Search Algorithms**: Understand and implement uninformed and informed search strategies
- **Machine Learning**: Build regression models from scratch with optimization algorithms
- **Optimization**: Apply local search techniques to combinatorial problems
- **Problem Solving**: Develop heuristic functions and analyze algorithm performance


