"""
Plot internationalisation (i18n) for every tracked run.

A ``--lang`` flag lets each run pick the language its plots are rendered in
(English or Ukrainian) via a ``--lang`` CLI flag surfaced in the web UI. Rather
than wrap every hardcoded label across the ~dozen plotting modules, we hook the
*save* step: ``install()`` monkeypatches ``matplotlib.figure.Figure.savefig`` so
that, when the active language is not English, every text artist on the figure
(title, axis labels, legend entries, categorical tick labels, annotations) is
translated in place just before the file is written. Numbers and untranslated
strings pass through unchanged, so coverage is incremental and safe.

Usage (in each run entrypoint)::

    import plot_i18n
    plot_i18n.set_lang(args.lang)   # 'en' (default) or 'uk'
    plot_i18n.install()             # no-op when lang == 'en'

This module lives at the repo root so the ``forecasting.*`` modules (run with
``python -m`` from the repo root) can import it.
"""
from __future__ import annotations

import os
from typing import Dict, List

# ── Active language ───────────────────────────────────────────────────────────

SUPPORTED = ('en', 'uk')
_DEFAULT = 'en'

# Initialise from the environment so a language set before import still applies
# (the per-dataset visualize runner sets this in-process). The CLI flag, when
# present, overrides via set_lang().
_current_lang = os.environ.get('PLOT_LANG', _DEFAULT).lower()
if _current_lang not in SUPPORTED:
    _current_lang = _DEFAULT


def set_lang(lang: str | None) -> None:
    """Set the active plot language. Unknown / None values fall back to English."""
    global _current_lang
    lang = (lang or _DEFAULT).lower()
    _current_lang = lang if lang in SUPPORTED else _DEFAULT
    # Mirror into the environment so a child process spawned in-process (the
    # per-dataset visualize runner uses runpy in the same interpreter, but a
    # subprocess fork would inherit this) sees the same language.
    os.environ['PLOT_LANG'] = _current_lang


def get_lang() -> str:
    return _current_lang


# ── Translation tables ────────────────────────────────────────────────────────
# EN → UK. Keys are the exact English strings used as plot labels. Translation
# is exact-match first; strings with no exact entry fall back to longest-first
# substring replacement using the phrase table, which handles runtime-formatted
# labels like "Daily profile — pv_electric_kw" or "Step 5 - Electric Demand".

_UK: Dict[str, str] = {
    # ── Time / calendar ──
    'Jan': 'Січ', 'Feb': 'Лют', 'Mar': 'Бер', 'Apr': 'Кві', 'May': 'Тра',
    'Jun': 'Чер', 'Jul': 'Лип', 'Aug': 'Сер', 'Sep': 'Вер', 'Oct': 'Жов',
    'Nov': 'Лис', 'Dec': 'Гру',
    'January': 'Січень', 'February': 'Лютий', 'March': 'Березень',
    'April': 'Квітень', 'June': 'Червень', 'July': 'Липень', 'August': 'Серпень',
    'September': 'Вересень', 'October': 'Жовтень', 'November': 'Листопад',
    'December': 'Грудень',
    'Mon': 'Пн', 'Tue': 'Вт', 'Wed': 'Ср', 'Thu': 'Чт', 'Fri': 'Пт',
    'Sat': 'Сб', 'Sun': 'Нд',
    'Monday': 'Понеділок', 'Tuesday': 'Вівторок', 'Wednesday': 'Середа',
    'Thursday': 'Четвер', 'Friday': "П'ятниця", 'Saturday': 'Субота',
    'Sunday': 'Неділя',
    'Weekday': 'Будній день', 'Weekend': 'Вихідні',
    'weekday': 'будній', 'weekend': 'вихідний',
    'Winter': 'Зима', 'Spring': 'Весна', 'Summer': 'Літо',
    'Autumn': 'Осінь', 'Fall': 'Осінь',
    'Date': 'Дата', 'Date / Hour': 'Дата / година', 'Month': 'Місяць',
    'Year': 'Рік', 'Hour of Day': 'Година доби', 'Hour of day': 'Година доби',
    'hour of day': 'година доби', 'Day of Year': 'День року',
    'Day of year': 'День року', 'Day in test period': 'День у тестовому періоді',

    # ── Generic axes / stats ──
    'Count': 'Кількість', 'Density': 'Щільність', 'Median': 'Медіана',
    'mean': 'середнє', 'max': 'макс', 'Mean': 'Середнє', 'Max': 'Макс',
    'Min': 'Мін', 'Std': 'Ст. відх.', 'Weight': 'Вага',
    'Pearson r': 'Кореляція Пірсона r', 'Skill score': 'Оцінка вправності',
    'Empirical CDF': 'Емпірична функція розподілу',
    'Fraction missing': 'Частка пропусків', 'Fraction open': 'Частка відкритих',
    'Image count': 'Кількість зображень',
    'Error band': 'Смуга похибки',

    # ── Units / physical quantities ──
    'kW': 'кВт', 'kWh/h': 'кВт·год/год', 'Power [kWh/h]': 'Потужність [кВт·год/год]',
    'Cost': 'Вартість', 'cost': 'вартість', 'Hourly cost': 'Погодинна вартість',
    'Temperature (°C)': 'Температура (°C)', 'Temperature (degC)': 'Температура (°C)',
    'Air temp (°C)': 'Темп. повітря (°C)', 'PV cell temp (°C)': 'Темп. комірки ПВ (°C)',
    'Fluid delta-T (°C)': 'Перепад темп. теплоносія (°C)',
    'Irradiance (W/m^2)': 'Опромінення (Вт/м²)',
    'Mean output (kW)': 'Середня потужність (кВт)',
    'Mean PV Output (kW)': 'Середня потужність ПВ (кВт)',
    'SOC [kWh]': 'Заряд [кВт·год]', 'Comfort [CU]': 'Комфорт [од.]',

    # ── Forecasting: series / models ──
    'Actual': 'Фактичне', 'Actual (7-day mean)': 'Фактичне (середнє за 7 днів)',
    'Predicted': 'Прогнозоване', 'Forecast': 'Прогноз', 'Context': 'Контекст',
    'Ensemble': 'Ансамбль', 'Adaptive': 'Адаптивний', 'adaptive': 'адаптивний',
    'Non-Adaptive': 'Неадаптивний', 'Non-adaptive': 'Неадаптивний',
    'non_adaptive': 'неадаптивний', 'Retrain': 'Перенавчання',
    'Combined': 'Комбіноване', 'Deficit': 'Дефіцит', 'Surplus': 'Надлишок',
    'Charge': 'Заряд', 'Discharge': 'Розряд',
    'Weekly-naive baseline': 'Тижнево-наївний базовий рівень',
    'Weekly-naive (MASE=1)': 'Тижнево-наївний (MASE=1)',
    'Worst base / window': 'Найгірша база / вікно',
    'Oracle (best base / window)': 'Оракул (найкраща база / вікно)',
    'Error calendar': 'Календар похибок',
    'Conditional MAE': 'Умовна MAE',
    'Per-model errors': 'Похибки за моделями',

    # ── Forecasting: axes ──
    'Forecast Hour': 'Година прогнозу', 'Forecast Window': 'Вікно прогнозу',
    'Window index': 'Індекс вікна', 'Forecast peak hour': 'Пікова година прогнозу',
    'Actual peak hour': 'Фактична пікова година',
    'Hour in forecast horizon': 'Година в горизонті прогнозу',
    'Hour of horizon': 'Година горизонту',
    'Evaluation Step': 'Крок оцінювання', 'Evaluation step': 'Крок оцінювання',
    'Overall MAE': 'Загальна MAE', 'Adaptive MAE': 'Адаптивна MAE',
    'Adaptive RMSE': 'Адаптивна RMSE', 'Non-adaptive MAE': 'Неадаптивна MAE',
    'Non-adaptive RMSE': 'Неадаптивна RMSE', 'Adaptive rolling MAE': 'Адаптивна ковзна MAE',
    'Adaptive PV rolling MAE': 'Адаптивна ковзна MAE для ПВ',
    'Mean MAE per Target': 'Середня MAE за цільовою змінною',
    'Mean MASE': 'Середня MASE',
    'Mean horizon-sum energy': 'Середня сумарна енергія за горизонт',
    'Aggregate forecast energy': 'Сукупна прогнозна енергія',
    'Day in forecast': 'День у прогнозі',

    # ── Forecasting: titles ──
    'Adaptive Forecaster Performance History':
        'Історія продуктивності адаптивного прогнозувальника',
    'Adaptive vs non-adaptive rolling forecast MAE':
        'Ковзна MAE прогнозу: адаптивний проти неадаптивного',
    'Summary: Adaptive vs Non-Adaptive':
        'Підсумок: адаптивний проти неадаптивного',
    'Overall MAE with Retrain Events':
        'Загальна MAE з подіями перенавчання',
    'Per-horizon MAE and RMSE': 'MAE та RMSE за горизонтом',
    'Per-horizon ensemble weights': 'Ваги ансамблю за горизонтом',
    'Ensemble win-rate by horizon step':
        'Частка перемог ансамблю за кроком горизонту',
    'Per-regime MAE — ensemble vs base learners':
        'MAE за режимом — ансамбль проти базових моделей',
    'Pooled predicted-vs-actual scatter':
        'Об’єднана діаграма прогноз–факт',
    'Residual distributions per target':
        'Розподіли залишків за цільовою змінною',
    'Ramp-rate error distributions':
        'Розподіли похибок швидкості зміни',
    'Daily peak events - forecast vs actual':
        'Добові пікові події — прогноз проти факту',
    'Daily peak events': 'Добові пікові події',
    'Load-duration curves - sorted actual vs forecast':
        'Криві тривалості навантаження — відсортовані факт проти прогнозу',
    'Consumption Forecast Overview — Test Period':
        'Огляд прогнозу споживання — тестовий період',
    'Generation Forecast Overview — Test Period':
        'Огляд прогнозу генерації — тестовий період',
    'PV Electric: Actual vs Predicted':
        'Електрична ПВ: факт проти прогнозу',
    'Cost of forecast error vs perfect-foresight oracle':
        'Вартість похибки прогнозу проти оракула ідеального передбачення',
    'Summary Metrics': 'Підсумкові метрики',
    'Electric Demand': 'Електричний попит', 'Thermal Demand': 'Тепловий попит',

    # ── Feature-impact titles ──
    'Generation  —  Feature Impact on PV Electric Output':
        'Генерація  —  вплив ознак на електричний вихід ПВ',
    'Generation  —  Feature Impact on Solar Thermal Output':
        'Генерація  —  вплив ознак на сонячний тепловий вихід',
    'Domestic Hot Water  —  Feature Impact':
        'Гаряче водопостачання  —  вплив ознак',
    'Feature Impact': 'Вплив ознак',

    # ── Dataset visualisation (generic) ──
    'Daily profile': 'Добовий профіль', 'Monthly profile': 'Місячний профіль',
    'Feature × target correlation': 'Кореляція ознака × ціль',
    'Top features': 'Найвпливовіші ознаки',
    'Missing Data Rate': 'Частка пропущених даних',
    'Annual Missing Fraction by Channel':
        'Річна частка пропусків за каналом',
    'Generation-Duration Curve': 'Крива тривалості генерації',
    'Generation-Duration Curves': 'Криві тривалості генерації',
    'Load-Duration Curves': 'Криві тривалості навантаження',
    'PV Generation-Duration Curve': 'Крива тривалості генерації ПВ',
    'Daily PV Capacity Factor': 'Добовий коефіцієнт використання ПВ',
    'Daily Capacity Factor': 'Добовий коефіцієнт використання',
    'Seasonal Diurnal PV Profile': 'Сезонний добовий профіль ПВ',
    'Seasonal Diurnal Profiles': 'Сезонні добові профілі',
    'Diurnal Profiles by Season': 'Добові профілі за сезоном',
    'Monthly Energy Balance': 'Місячний енергетичний баланс',
    'Monthly Energy Balance -- Demand vs Generation':
        'Місячний енергобаланс — попит проти генерації',
    'Decade Comparison — First vs Second 10 Years':
        'Порівняння десятиліть — перші проти других 10 років',
    'Sky-class counts by month': 'Кількість класів неба за місяцем',
    'Occupancy Pattern -- Day-of-Year x Hour Heatmap':
        'Шаблон заповненості — теплокарта день року × година',
    'HVAC Electric Demand vs Outdoor Temperature':
        'Електроспоживання ОВК проти зовнішньої температури',
    'Thermal Demand vs Outdoor Temperature':
        'Тепловий попит проти зовнішньої температури',
    'PV Generation vs Global Tilted Irradiance':
        'Генерація ПВ проти глобального похилого опромінення',

    # ── Optimization ──
    'Domestic demand': 'Побутовий попит', 'Thermal demand': 'Тепловий попит',
    'Domestic served': 'Покритий побутовий', 'Thermal served': 'Покритий тепловий',
    'Domestic demand (actual)': 'Побутовий попит (факт)',
    'Domestic demand (forecast)': 'Побутовий попит (прогноз)',
    'Thermal demand (actual)': 'Тепловий попит (факт)',
    'Thermal demand (forecast)': 'Тепловий попит (прогноз)',
    'PV (forecast)': 'ПВ (прогноз)', 'PV (kW)': 'ПВ (кВт)',
    'PV (actual)': 'ПВ (факт)', 'PV electric': 'Електрична ПВ',
    'SC (forecast)': 'СК (прогноз)', 'Diesel cost': 'Вартість дизеля',
    'Passive solar credit': 'Пасивний сонячний внесок',
    'Renewable share [%]': 'Частка ВДЕ [%]', 'REF — renewable share [%]': 'ВДЕ — частка ВДЕ [%]',
    'Self-consumption [%]': 'Самоспоживання [%]',
    'Demand coverage [%]': 'Покриття попиту [%]',
    'Comfort units and hourly cost': 'Одиниці комфорту та погодинна вартість',
    'Cumulative operating cost per mode':
        'Накопичена експлуатаційна вартість за режимом',
    'Cumulative cost over oracle': 'Накопичена вартість понад оракул',
    'Cumulative penalty  [UAH]': 'Накопичений штраф [грн]',
    'Per-window penalty  [UAH]': 'Штраф за вікно [грн]',
    'Mode comparison — aggregated KPIs':
        'Порівняння режимів — агреговані KPI',
    'Electric battery state of charge': 'Заряд електричної батареї',
    'Thermal buffer state of charge': 'Заряд теплового буфера',
    'Electric bus — supply vs demand': 'Електрична шина — пропозиція проти попиту',
    'Thermal bus — supply vs demand': 'Теплова шина — пропозиція проти попиту',
    'Domestic comfort — coverage breakdown':
        'Побутовий комфорт — розподіл покриття',
    'Thermal comfort — coverage breakdown':
        'Тепловий комфорт — розподіл покриття',
    'Hourly self-consumption of renewable generation':
        'Погодинне самоспоживання відновлюваної генерації',
    'Ensemble vs oracle': 'Ансамбль проти оракула',
    'Daily-energy scatter': 'Діаграма добової енергії',

    # ── Misc shared tokens ──
    'GBDT': 'GBDT', 'LSTM': 'LSTM', 'LSTM-CNN': 'LSTM-CNN', 'TFT': 'TFT',
    'MASE': 'MASE', 'Mean MAE': 'Середня MAE',
    'rolling': 'ковзне', 'Step': 'Крок',
    'Renewable share': 'Частка ВДЕ',
}

# Short / ambiguous keys that are safe as full-string matches but MUST NOT be
# used for substring replacement (they would corrupt longer labels).
_EXACT_ONLY = {
    'kW', 'max', 'mean', 'Max', 'Min', 'Mean', 'Std', 'cost', 'Cost',
    'GBDT', 'LSTM', 'TFT', 'MASE', 'Step', 'rolling', 'Year', 'Month', 'Date',
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
    'Nov', 'Dec', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun',
    'adaptive', 'weekday', 'weekend',
}

# Phrase table for substring fallback: every entry except the exact-only set,
# sorted longest-first so the most specific phrase wins.
_PHRASES: List[tuple] = sorted(
    ((k, v) for k, v in _UK.items() if k not in _EXACT_ONLY),
    key=lambda kv: len(kv[0]), reverse=True,
)


def translate(text: str) -> str:
    """Translate a single label into the active language.

    Exact match first (highest quality); otherwise longest-first substring
    replacement so runtime-formatted labels ("Daily profile — pv_electric_kw")
    still get their static parts translated. Unknown text passes through.
    """
    if _current_lang == 'en' or not text:
        return text
    table = _UK
    hit = table.get(text)
    if hit is not None:
        return hit
    out = text
    for en, uk in _PHRASES:
        if en in out:
            out = out.replace(en, uk)
    return out


# ── matplotlib hook ───────────────────────────────────────────────────────────

_installed = False
_orig_savefig = None


def translate_figure(fig) -> None:
    """Translate every text artist on a figure in place (current language)."""
    if _current_lang == 'en':
        return
    import matplotlib.text as mtext
    for artist in fig.findobj(mtext.Text):
        # Skip already-translated artists so a second savefig on the same
        # figure doesn't double-process.
        if getattr(artist, '_i18n_done', False):
            continue
        try:
            s = artist.get_text()
        except Exception:
            continue
        if not s:
            continue
        new = translate(s)
        if new != s:
            artist.set_text(new)
        artist._i18n_done = True


def install() -> None:
    """Monkeypatch Figure.savefig to translate the figure first.

    No-op when the active language is English, and idempotent. Call once per
    process after set_lang().
    """
    global _installed, _orig_savefig
    if _installed or _current_lang == 'en':
        return
    from matplotlib.figure import Figure
    _orig_savefig = Figure.savefig

    def _savefig_i18n(self, *args, **kwargs):
        try:
            translate_figure(self)
        except Exception:
            pass  # translation is best-effort; never break a save
        return _orig_savefig(self, *args, **kwargs)

    Figure.savefig = _savefig_i18n  # type: ignore[assignment]
    _installed = True


def add_arg(parser) -> None:
    """Register the shared --lang flag on an argparse parser."""
    parser.add_argument(
        '--lang', choices=list(SUPPORTED), default=_DEFAULT,
        help='Language for generated plot labels (en=English, uk=Ukrainian).',
    )


def setup_from_args(args) -> None:
    """Convenience: set_lang(args.lang) + install() in one call."""
    set_lang(getattr(args, 'lang', None))
    install()


# Auto-install when a non-English language was set in the environment before this
# module was imported. This covers subprocesses spawned by an orchestrator that
# already called set_lang() (which exports PLOT_LANG): they inherit the env and
# get the savefig hook just by importing a plotting module that imports us.
if _current_lang != 'en':
    install()
