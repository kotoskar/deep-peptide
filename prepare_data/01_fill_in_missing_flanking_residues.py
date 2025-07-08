import csv

def fix_region(region: str, is_pre: bool) -> str:
    """
    Корректирует поле pre-region или post-region согласно правилам:
    - Если пустое, возвращает '@@'
    - Если один символ, добавляет '@' слева (для pre-region) или справа (для post-region)
    - Если два символа, возвращает как есть
    """
    if not region or region.strip() == '':
        return '@@'
    region = region.strip()
    if len(region) == 1:
        if is_pre:
            return '@' + region
        else:
            return region + '@'
    if len(region) == 2:
        return region
    # Если длина больше 2, можно оставить как есть или обрезать (зависит от задачи)
    return region

input_file = 'flanking_regions.csv'    # замените на имя вашего файла
output_file = 'flanking_regions_no_missing.csv'

with open(input_file, newline='', encoding='utf-8') as csv_in, \
     open(output_file, 'w', newline='', encoding='utf-8') as csv_out:
    
    reader = csv.DictReader(csv_in)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
    writer.writeheader()
    
    for row in reader:
        row['pre-region'] = fix_region(row.get('pre-region', ''), is_pre=True)
        row['post-region'] = fix_region(row.get('post-region', ''), is_pre=False)
        writer.writerow(row)

print(f"Обработка завершена. Результат сохранён в {output_file}")
