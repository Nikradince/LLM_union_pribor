"""Validation and deterministic rendering. No network or filesystem side effects."""

import json
import re
from dataclasses import dataclass


class ValidationError(ValueError):
    """Input or model response violates the generation contract."""


class InsufficientDataError(ValidationError):
    """More verified source material is needed to produce the requested draft."""


def text(value: object, field: str, limit: int = 5000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValidationError(f"{field}: нужна непустая строка длиной до {limit}")
    return value.strip()


def parse_json(payload: str | bytes) -> object:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"Повторяющееся поле JSON: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValidationError("NaN и Infinity не допускаются в JSON")

    return json.loads(payload, object_pairs_hook=unique_object, parse_constant=reject_constant)


@dataclass(frozen=True)
class Product:
    name: str
    manufacturer: str
    specs: dict[str, str]
    description: str
    required_specs: list[str]
    source_url: str
    source_notes: list[str]
    product_type: str = ""
    audience: str = ""
    tasks: tuple[str, ...] = ()

    @classmethod
    def parse(cls, raw: object) -> "Product":
        if not isinstance(raw, dict):
            raise ValidationError("Товар должен быть JSON-объектом")
        name = text(raw.get("name"), "name", 300)
        manufacturer = raw.get("manufacturer")
        if manufacturer is None or isinstance(manufacturer, str) and not manufacturer.strip():
            manufacturer = "данных нет"
        else:
            manufacturer = text(manufacturer, "manufacturer", 300)
        specs = raw.get("specs", {})
        if specs is None:
            specs = {}
        if not isinstance(specs, dict) or len(specs) > 50:
            raise ValidationError("specs: нужен объект, не более 50 характеристик")
        clean_specs = {}
        for key, value in specs.items():
            key = text(key, "specs.key", 100)
            if key in clean_specs:
                raise ValidationError("Названия характеристик совпадают после удаления пробелов")
            clean_specs[key] = (
                "данных нет" if value is None or isinstance(value, str) and not value.strip()
                else text(value, key, 500)
            )
        description = raw.get("description", "")
        if description is None:
            description = ""
        if not isinstance(description, str) or len(description) > 15000:
            raise ValidationError("description: нужна строка длиной до 15000")
        required = raw.get("required_specs", [])
        notes = raw.get("source_notes", [])
        normalized = []
        for field, values in (("required_specs", required), ("source_notes", notes)):
            if not isinstance(values, list) or len(values) > 50:
                raise ValidationError(f"{field}: нужен список не более 50 строк")
            normalized.append(list(dict.fromkeys(text(item, field, 1000) for item in values)))
        required, notes = normalized
        url = raw.get("source_url", "")
        if not isinstance(url, str) or len(url) > 2000:
            raise ValidationError("source_url: нужна строка длиной до 2000")
        kind = raw.get("product_type", "")
        audience = raw.get("audience", "")
        for field, value in (("product_type", kind), ("audience", audience)):
            if not isinstance(value, str) or len(value) > 300:
                raise ValidationError(f"{field}: нужна строка до 300 символов")
        tasks = raw.get("tasks", [])
        if not isinstance(tasks, list) or len(tasks) > 5:
            raise ValidationError("tasks: нужен список до 5 исходных формулировок задач")
        tasks = tuple(text(t, "tasks", 170) for t in tasks)
        return cls(name, manufacturer, clean_specs, description.strip(), required, url, notes,
                   kind.strip(), audience.strip(), tasks)

    def characteristics(self) -> dict[str, str]:
        return {**dict.fromkeys(self.required_specs, "данных нет"), **self.specs}

    def missing_fields(self) -> list[str]:
        missing = []
        if self.manufacturer == "данных нет":
            missing.append("manufacturer")
        if not self.description:
            missing.append("description")
        if not self.specs:
            missing.append("specs")
        if not self.required_specs:
            missing.append("required_specs: список ожидаемых полей не задан; полнота не проверена")
        if not self.audience:
            missing.append("audience")
        if not self.tasks:
            missing.append("tasks")
        missing.extend(k for k, v in self.characteristics().items() if v == "данных нет")
        return list(dict.fromkeys(missing))


def load_records(payload: str) -> list:
    try:
        raw = parse_json(payload)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Некорректный JSON: строка {exc.lineno}, столбец {exc.colno}") from None
    except RecursionError:
        raise ValidationError("JSON содержит слишком много уровней вложенности") from None
    raw = raw if isinstance(raw, list) else [raw]
    if not 1 <= len(raw) <= 20:
        raise ValidationError("Ожидается от 1 до 20 товаров")
    return raw


def load_products(payload: str) -> list[Product]:
    return [Product.parse(item) for item in load_records(payload)]


def block_catalog(product: Product) -> dict[str, str]:
    """Every output sentence must be a source fragment or a reviewed neutral template."""
    blocks = {"name": f"{product.name}."}
    if product.manufacturer != "данных нет":
        blocks["manufacturer"] = f"Производитель: {product.manufacturer}."
    # Split only at sentence boundaries; decimal points and units remain untouched.
    fragments = []
    for line in product.description.splitlines():
        fragments.extend(re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z])", line.strip()))
    for i, fragment in enumerate(fragments):
        if fragment:
            # Normalize list punctuation, preserving every technical value.
            fragment = fragment.rstrip("; ")
            if not fragment:
                continue
            if fragment[-1] not in ".!?":
                fragment += "."
            blocks[f"source_{i}"] = fragment[0].upper() + fragment[1:]
    source_tokens = [
        re.findall(r"[-+]?\d+(?:[.,]\d+)?|\w+", value.casefold())
        for key, value in blocks.items() if key.startswith("source_")
    ]
    for i, (key, value) in enumerate(product.specs.items()):
        if value == "данных нет":
            continue
        tokens = re.findall(r"[-+]?\d+(?:[.,]\d+)?|\w+", value.casefold())

        def covered_by_source(source: list[str]) -> bool:
            # Ordered matching tolerates extra qualifiers in the source sentence.
            remaining = iter(source)
            return bool(tokens) and all(token in remaining for token in tokens)

        if not any(covered_by_source(source) for source in source_tokens):
            blocks[f"spec_{i}"] = f"{key}: {value}."
    blocks["editorial_0"] = (
        "Перед заказом сопоставьте приведенные сведения с требованиями вашего проекта."
    )
    blocks["editorial_1"] = (
        "Совместимость с конкретным оборудованием необходимо подтвердить по документации."
    )
    blocks["editorial_2"] = (
        "Условия поставки и состав комплектации следует уточнить у поставщика."
    )
    return blocks


def candidates(product: Product) -> tuple[list[str], list[str]]:
    # Truncation is explicit: the full identity remains in product_name.
    title = product.name[:70].rstrip()
    titles = [title]
    combined = f"{product.manufacturer} {product.name}"
    if product.manufacturer != "данных нет" and len(combined) <= 70:
        titles.append(combined)
    if product.product_type:
        vendor = product.manufacturer if product.manufacturer != "данных нет" else ""
        descriptive = " ".join(filter(None, [product.product_type, vendor, product.name]))
        if len(descriptive) <= 70:
            titles.insert(0, descriptive)
    queries = [product.name, f"{product.name} купить", f"{product.name} характеристики"]
    if product.manufacturer != "данных нет":
        queries.append(f"{product.manufacturer} {product.name}")
    if product.product_type:
        vendor = product.manufacturer if product.manufacturer != "данных нет" else ""
        queries = [f"{product.product_type} {product.name}",
                   " ".join(filter(None, [product.product_type, vendor, "купить"])),
                   f"{product.name} характеристики", *queries]
    return titles, list(dict.fromkeys(queries))


def render_card(product: Product, plan: object) -> dict:
    if not isinstance(plan, dict) or set(plan) != {
        "title_index", "description_blocks", "commercial_blocks", "query_indices"
    }:
        raise ValidationError("План должен содержать только четыре поля из схемы")
    blocks = block_catalog(product)
    blocks.update({f"task_{i}": task for i, task in enumerate(product.tasks)})
    titles, queries = candidates(product)

    def select_indices(field: str, values: list[str], low: int, high: int) -> list[str]:
        ids = plan.get(field)
        if not isinstance(ids, list) or not low <= len(ids) <= high:
            raise ValidationError(f"{field}: требуется от {low} до {high} индексов")
        if any(type(i) is not int or not 0 <= i < len(values) for i in ids):
            raise ValidationError(f"{field}: неизвестный индекс")
        if len(set(ids)) != len(ids):
            raise ValidationError(f"{field}: повторяющиеся индексы")
        return [values[i] for i in ids]

    def select_blocks(field: str, low: int, high: int) -> tuple[str, list[str]]:
        ids = plan.get(field)
        if not isinstance(ids, list) or not low <= len(ids) <= high:
            raise ValidationError(f"{field}: требуется от {low} до {high} блоков")
        if any(not isinstance(i, str) or i not in blocks for i in ids):
            raise ValidationError(f"{field}: неизвестный идентификатор блока")
        if len(set(ids)) != len(ids):
            raise ValidationError(f"{field}: повторяющиеся блоки")
        return " ".join(blocks[i] for i in ids), ids

    title_id = plan.get("title_index")
    if type(title_id) is not int or not 0 <= title_id < len(titles):
        raise ValidationError("title_index: неизвестный индекс")
    description, desc_ids = select_blocks("description_blocks", 1, 30)
    if any(i.startswith("task_") for i in desc_ids):
        raise ValidationError("Для SEO-описания используйте source/spec-блоки, задачи — только для commercial")
    if not 800 <= len(description) <= 1200:
        raise ValidationError(f"SEO-описание: {len(description)} символов, требуется 800-1200")
    if not any(i.startswith("source_") or i.startswith("spec_") for i in desc_ids):
        raise ValidationError("SEO-описание должно содержать исходные факты")
    commercial, commercial_ids = select_blocks("commercial_blocks", 0, 3)
    if any(not i.startswith("task_") for i in commercial_ids):
        raise ValidationError("Commercial-блок: выбирайте только task-блоки")
    if product.tasks and not commercial_ids:
        raise ValidationError("Commercial-блок: выберите хотя бы одну исходную задачу")
    audience = product.audience or "данных нет; уточнить целевую аудиторию"
    tasks = [blocks[i] for i in commercial_ids]
    commercial = f"Кому: {audience}. Задачи: " + (commercial or "данных нет; уточнить назначение")
    if len(commercial) > 600:
        raise ValidationError("Commercial-блок: не более 600 символов")
    selected_queries = select_indices("query_indices", queries, 3, 5)
    missing = product.missing_fields()
    return {
        "product_name": product.name,
        "status": "draft_requires_review",
        "seo_title": titles[title_id],
        "seo_description": description,
        "characteristics": {"Производитель": product.manufacturer, **product.characteristics()},
        "search_queries": selected_queries,
        "commercial": commercial,
        "commercial_details": {"audience": audience, "tasks": tasks},
        "missing_fields": missing,
        "warnings": [
            "Черновик: перед публикацией нужна проверка контент-менеджером.",
            *(["Недостающие сведения: " + ", ".join(missing)] if missing else []),
            *product.source_notes,
        ],
        "source_url": product.source_url,
        "evidence": {"description_blocks": desc_ids, "commercial_blocks": commercial_ids},
    }


def markdown(results: list[dict]) -> str:
    def escape(value: str) -> str:
        return re.sub(r"([\\`*_{}\[\]<>#|])", r"\\\1", value)

    sections = []
    for item in results:
        if "error" in item:
            missing = ", ".join(item.get("missing_fields", []))
            sections.append(f"## Товар {item['index'] + 1}\n\nОшибка: {escape(item['error'])}"
                            + (f"\n\nУточнить: {escape(missing)}" if missing else ""))
            continue
        card = item["card"]
        specs = "\n".join(f"- {escape(k)}: {escape(v)}" for k, v in card["characteristics"].items())
        queries = "\n".join(f"- {escape(q)}" for q in card["search_queries"])
        warnings = "\n".join(f"- {escape(w)}" for w in card["warnings"])
        details = card["commercial_details"]
        tasks = "\n".join(f"- {escape(t)}" for t in details["tasks"]) or "- данных нет"
        sections.append(
            f"# {escape(card['seo_title'])}\n\n{escape(card['seo_description'])}\n\n"
            f"## Характеристики\n\n{specs}\n\n## Поисковые запросы\n\n{queries}\n\n"
            f"## Для менеджера\n\nКому: {escape(details['audience'])}.\n\nЗадачи:\n\n{tasks}"
            f"\n\n## Проверить\n\n{warnings}\n\n"
            f"Источник: {escape(card['source_url']) or 'не указан'}"
        )
    return "\n\n---\n\n".join(sections) + "\n"
