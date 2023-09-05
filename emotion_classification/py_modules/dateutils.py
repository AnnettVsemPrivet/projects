def count_months(start_date, end_date):
    if end_date < start_date:  # This means we only support datetime, not date
        raise
    full_years = (end_date.year - start_date.year) - 1
    start_year_months = 13 - start_date.month
    return start_year_months + (full_years * 12) + end_date.month


def get_week_of_month(date):
    first_day_weekday = date.replace(day=1).weekday()
    week_of_month = int((first_day_weekday-1 + date.day) / 7)
    return week_of_month