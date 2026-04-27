import pandas as pd
import numpy as np

# 1. Đọc dữ liệu
BASE_DATA = "data/"
orders = pd.read_csv(BASE_DATA + 'orders.csv', parse_dates=['order_date'])
order_items = pd.read_csv(BASE_DATA + 'order_items.csv')
products = pd.read_csv(BASE_DATA + 'products.csv')

# 2. Merge các bảng cần thiết
# Nối order_items với orders để lấy order_date
df = order_items.merge(orders[['order_id', 'order_date', 'order_status']], on='order_id', how='inner')
# Nối tiếp với products để lấy cogs (giá vốn)
df = df.merge(products[['product_id', 'cogs']], on='product_id', how='inner')

# Lọc các đơn hàng không bị hủy
df = df[df['order_status'] != 'cancelled']

# 3. Tính Doanh Thu (Revenue) và Giá vốn thực tế (Actual COGS)
df['revenue'] = (df['quantity'] * df['unit_price']) - df['discount_amount'].fillna(0)
df['total_actual_cogs'] = df['quantity'] * df['cogs']

# 4. Xác định phân loại năm (Năm chẵn / Năm lẻ)
df['year'] = df['order_date'].dt.year
df['year_type'] = np.where(df['year'] % 2 == 0, 'Even_Year (Năm chẵn)', 'Odd_Year (Năm lẻ)')

# 5. Tổng hợp để tính Margin Rate
summary = df.groupby('year_type')[['revenue', 'total_actual_cogs']].sum()
summary['margin'] = summary['revenue'] - summary['total_actual_cogs']
summary['margin_rate'] = summary['margin'] / summary['revenue']

print("="*60)
print("PHÂN TÍCH MARGIN RATE THEO NĂM CHẴN VÀ NĂM LẺ TỪ DỮ LIỆU GỐC")
print("="*60)
print(summary[['revenue', 'margin', 'margin_rate']])

# Trích xuất các hệ số Margin Rate ra biến độc lập (Scalar)
even_year_margin_rate = summary.loc['Even_Year (Năm chẵn)', 'margin_rate'] if 'Even_Year (Năm chẵn)' in summary.index else 0
odd_year_margin_rate = summary.loc['Odd_Year (Năm lẻ)', 'margin_rate'] if 'Odd_Year (Năm lẻ)' in summary.index else 0

print("\n=> Hệ số Margin Rate - Năm Chẵn : {:.2%}".format(even_year_margin_rate))
print("=> Hệ số Margin Rate - Năm Lẻ   : {:.2%}".format(odd_year_margin_rate))


# ==========================================
# ỨNG DỤNG: Hàm nội suy ngược COGS từ Revenue
# ==========================================
def deduce_cogs(revenue, revenue_year):
    """
    Suy ngược Giá vốn (COGS) từ Doanh thu của một năm
    Quy tắc: 
        Margin Rate = (Revenue - COGS) / Revenue
    =>  COGS = Revenue * (1 - Margin Rate)
    """
    if revenue_year % 2 == 0:
        return revenue * (1 - even_year_margin_rate)
    else:
        return revenue * (1 - odd_year_margin_rate)

print("\n" + "="*60)
print("VÍ DỤ ƯỚC TÍNH NGƯỢC COGS TỪ REVENUE (Áp dụng hàm deduce_cogs)")
print("="*60)
sample_revenue = 5000000

# Thử nghiệm với một năm chẵn 
test_year_even = 2022
estimated_cogs_even = deduce_cogs(sample_revenue, test_year_even)
print(f"Nếu Doanh Thu năm {test_year_even} là {sample_revenue:,} USD")
print(f" -> COGS nội suy: {estimated_cogs_even:,.2f} USD")

# Thử nghiệm với một năm lẻ 
test_year_odd = 2025
estimated_cogs_odd = deduce_cogs(sample_revenue, test_year_odd)
print(f"\nNếu Doanh Thu năm {test_year_odd} là {sample_revenue:,} USD")
print(f" -> COGS nội suy: {estimated_cogs_odd:,.2f} USD")
