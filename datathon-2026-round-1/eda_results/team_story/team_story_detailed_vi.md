# Bao Cao Chi Tiet Dataset + Model (Tieng Viet)

Thoi diem tao: 2026-04-27 13:42 UTC

Phan nay la ban mo rong chi tiet, dung de trao doi voi team business + data + operations.
Toan bo so lieu duoi day duoc lay tu cac bang trong `eda_results/team_story/tables`.

## 1) Tong quan lanh dao (Executive Brief)

1. Quy mo doanh thu toan ky dat **16,430,476,586 VND**, COGS **14,163,450,519 VND**, loi nhuan gop **2,267,026,066 VND**.
2. Du lieu co tinh mua vu ro: doanh thu trung binh theo thu trong tuan chenh lech manh (cao nhat Thu Tu ~**4.68M/ngay**, thap nhat Thu Bay ~**3.91M/ngay**).
3. Ap luc discount la diem nghen lon nhat: bucket discount cao co margin am, dac biet bucket **20%+** co margin **-24.56%**.
4. RUI RO van hanh hien huu: payment method `cod` co cancellation rate cao nhat (**16.00%**), ly do return lon nhat la `wrong_size` (**13,967** don vi return).
5. Mo hinh du bao hien tai van on dinh nhat o nhom `core_long` (Rev MAE **988,549**). Nhieu submission tu repo khac vi pham rang buoc `COGS < Revenue` neu dung nguyen ban.

## 2) Cau truc du lieu va pham vi phan tich

- Nguon su dung:
  - Sales daily: `sales.csv`
  - Giao dich: `orders.csv`, `order_items.csv`, `payments.csv`, `returns.csv`, `shipments.csv`
  - Master data: `products.csv`, `customers.csv`, `geography.csv`, `promotions.csv`
  - Van hanh/traffic: `inventory.csv`, `web_traffic.csv`
- Fact line duoc dung de phan tich economics:
  - `gross_revenue = quantity * unit_price`
  - `net_revenue = gross_revenue - discount_amount`
  - `line_cogs = quantity * cogs`
  - `gross_profit = net_revenue - line_cogs`
  - `discount_rate = discount_amount / gross_revenue`

## 3) Story doanh thu va bien loi nhuan theo thoi gian

Xem hinh:
- `fig01_yearly_financials.png`
- `fig02_monthly_heatmap.png`
- `fig03_weekday_pattern.png`

So lieu chinh:
- Nam doanh thu cao nhat: **2016** voi **2,104,640,678 VND**.
- Nam doanh thu thap nhat: **2012** voi **741,497,748 VND**.
- CAGR full period (theo yearly aggregate): **+4.66%**.
- Bien loi nhuan gop theo nam dao dong lon: nam tot nhat ~**20.77%**, nam thap nhat ~**9.77%**.

Dien giai business:
- Planning theo nam khong du, can planning theo thang/quy vi seasonality manh.
- Team finance khong nen dung 1 margin target co dinh cho ca nam; nen dung margin corridor theo mua.

## 4) Demand pattern theo tuan va lich

Xem hinh:
- `fig03_weekday_pattern.png`

Chi tiet:
- Thu doanh thu cao nhat: `Wed` ~**4,680,065 VND/ngay**.
- Thu doanh thu thap nhat: `Sat` ~**3,906,581 VND/ngay**.
- Khoang cach giua ngay cao nhat va thap nhat xap xi **19.8%**.

Y nghia van hanh:
- Co the doi lich campaign va push SKU margin cao vao Tue/Wed/Thu.
- Sat/Sun phu hop hon cho inventory-clearing hoac conversion campaign ngan han.

## 5) Story category va segment profitability

Xem hinh:
- `fig04_category_scatter.png`
- `fig05_segment_profitability.png`

Category:
- `Streetwear`: net revenue **12,558,477,099**, margin **9.28%** (lon nhat ve quy mo, nhung margin khong cao).
- `Outdoor`: net revenue **2,353,396,797**, margin **11.35%**.
- `Casual`: net revenue **440,285,194**, margin **7.66%** (thap nhat trong cac category lon).

Segment margin top:
- `Trendy`: **15.47%**
- `Activewear`: **12.76%**
- `Standard`: **12.72%**

Ham y:
- Category doanh thu lon nhat chua chac la category hieu qua nhat.
- Neu muc tieu la toi uu loi nhuan, can tach KPI doanh thu va KPI margin theo category/segment.

## 6) Story discount va promotion economics

Xem hinh:
- `fig06_discount_buckets.png`
- `fig07_promo_type.png`

KQ discount bucket:
- `0%`: 438,353 lines, margin **19.96%**, net revenue **10,995,039,053**.
- `0-5%`: margin **-63.28%**.
- `5-10%`: margin **0.87%**.
- `10-20%`: margin **-10.85%**.
- `20%+`: margin **-24.56%**.

KQ promo type:
- `percentage`: net revenue **4,314,083,021**, gross profit **-442,339,348** (margin **-10.25%**).
- `fixed`: net revenue **371,747,191**, gross profit **-235,257,291** (margin **-63.28%**).

Ket luan business quan trong:
- Discount dang mua doanh thu bang loi nhuan.
- Can bo sung "promo profitability gate": campaign nao margin sau promo am phai duoc canh bao/suspended.

## 7) Story returns, cancellation, leakage

Xem hinh:
- `fig08_returns_reasons.png`
- `fig09_cancellation_rates.png`
- `fig13_inventory_stress.png`

Returns:
- Top ly do:
  - `wrong_size`: **13,967**
  - `defective`: **8,020**
  - `not_as_described`: **7,035**

Cancellation theo payment:
- Cao nhat theo rate: `cod` **16.00%**.
- Cao nhat theo volume: `credit_card` **28,452** cancelled orders.

Cancellation theo source:
- `referral` **9.34%**
- `direct` **9.28%**
- `organic_search` **9.23%**

Inventory leakage:
- Thang stockout cao nhat: **2015-11** voi stockout rate **72.54%**.
- Thang fill rate thap nhat: **2013-12** voi **94.38%**.

Ham y:
- Team ops can co dashboard canh bao som theo stockout-rate by month.
- Team product/merchandising can uu tien giai quyet fit/size issue de cat return reason `wrong_size`.

## 8) Story customer, cohort, region, traffic

Xem hinh:
- `fig10_customer_cohorts.png`
- `fig11_age_group_orders.png`
- `fig12_region_trend.png`
- `fig14_web_traffic_corr.png`

Cohort:
- Peak acquisition month: **2022-12**, them **1,883** khach hang.
- Cumulative customers cuoi ky: **121,930**.

Age group:
- Orders per customer cao nhat: `55+` = **7.27**.
- Cac nhom khac cung cao va gan nhau (7.07 -> 7.22), cho thay retention base kha dong deu.

Region:
- Tong doanh thu proxy toan ky:
  - `East`: **7,291,150,819**
  - `Central`: **4,719,491,268**
  - `West`: **3,670,227,178**
- Nam moi nhat, `East` van dan dau (**520,816,706**).

Traffic:
- Corr(sessions, revenue) = **0.321** (muc vua).
- Corr(sessions, orders) = **0.191** (yeu-vua).
- Corr(bounce_rate, revenue) = **-0.021** (gan nhu khong).

Ham y:
- Tang sessions khong dong nghia tang revenue tuong ung; can toi uu quality traffic + conversion funnel.

## 9) Danh gia model hien tai (main repo)

Xem hinh:
- `fig15_feature_set_performance.png`
- `fig16_feature_importance_top20.png`

Feature set ranking:
- Tot nhat: `core_long`
  - Rev MAE: **988,549**
  - Rev RMSE: **1,409,008**
  - Rev R2: **0.3425**
  - COGS MAE: **869,556**
  - COGS RMSE: **1,234,531**
  - COGS R2: **0.3269**
- Kem nhat: `core_long_plus_biz730`
  - Rev MAE tang **+10.80%** so voi best.

Top feature Revenue (gain):
- `yoy_365_730`, `lag_1095`, `lag_366`, `lag_364`, `lag_730`, `lag_365`, `roll7_std`, `roll7_mean`, `trend_pred`.

Y nghia:
- Model phu thuoc manh vao long-seasonal lags + ratio dai han + trend.
- Viec them business lag 730 khong mac dinh giup, co the do nhiu/noise va mismatch signal.

## 10) Danh gia solution tu repo khac: hop le hay khong

Xem hinh:
- `fig17_candidate_comparison.png`
- `fig18_candidate_fix_impact.png`

Phat hien quan trong:
- Nhieu external submission bi vi pham rang buoc `COGS < Revenue` truoc khi sua:
  - `r1_residual_ensemble`: **159** dong vi pham
  - `r1_lgb`: **102**
  - `datathon2026_submission`: **102**
  - `r1_ensemble`: **71**
- Sau khi fix rang buoc, tat ca candidate trong `cross_repo_candidates` deu compliance.

So sanh profile du bao voi baseline 2022:
- `ours_current_fixed`: Rev mean **-1.76%** vs 2022 (safe profile).
- `r1_residual_ensemble_fixed`: **+0.19%** (balanced profile).
- `submission_blend_B`: **+19.35%** (medium aggressive).
- `r1_ensemble_fixed`: **+38.50%** (aggressive profile).

Khuyen nghi su dung theo muc do rui ro:
1. Safe: `ours_current_fixed.csv`
2. Balanced: `r1_residual_ensemble_fixed.csv`
3. Medium: `submission_blend_B.csv`
4. Aggressive: `submission_blend_A.csv` / `r1_ensemble_fixed.csv`

## 11) Cac phat hien can team tranh luan ky

1. Discount policy:
   - Co nen dat tran discount theo category (vi margin am o bucket cao)?
2. Returns policy:
   - Thu nghiem gi de giam `wrong_size` nhanh nhat (size guide, fit recommendation, PDP copy)?
3. Payment funnel:
   - Vi sao `cod` co cancel rate cao gap doi mat bang? Co can thay doi UX checkout cho COD?
4. Ops policy:
   - Nguong stockout-rate nao se trigger campaign hold?
5. Forecast policy:
   - Chon profile safe hay balanced cho planning Q3/Q4?

## 12) De xuat action 30-60-90 ngay

30 ngay:
- Lap dashboard leakage core: discount bucket margin, cancel by payment/source, return reason, stockout.
- Dat guardrail: campaign nao projected margin < 0 thi can approval cap cao.

60 ngay:
- Chay A/B giam return `wrong_size` tren top category.
- Toi uu checkout cho `cod` + source co cancel cao.

90 ngay:
- Chuan hoa cycle planning theo seasonality + region.
- Tich hop model plausibility checks vao quy trinh submit forecast (constraint + growth sanity + risk label).

## 13) Danh muc tep de dung trong buoi team

- Bao cao nay:
  - `eda_results/team_story/team_story_detailed_vi.md`
- Figure deck (18 hinh):
  - `eda_results/team_story/figures/`
- Bang so lieu chi tiet (21 bang):
  - `eda_results/team_story/tables/`
- Ban doi chieu cross-repo:
  - `eda_results/cross_repo_audit/README.md`

