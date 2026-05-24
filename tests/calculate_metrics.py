import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Попытка импорта быстрой библиотеки для расстояния Левенштейна
try:
    import Levenshtein
    def edit_distance(s1, s2):
        return Levenshtein.distance(s1, s2)
except ImportError:
    from difflib import SequenceMatcher
    def edit_distance(s1, s2):
        # difflib возвращает similarity (0..1), конвертируем в distance
        return int(len(s1) * (1 - SequenceMatcher(None, s1, s2).ratio()))

# Импортируем вашу функцию выполнения запросов
try:
    from src.onec.integration.http_request import executeQuery
except ImportError as e:
    print(f"Ошибка импорта executeQuery: {e}")
    print("Убедитесь, что скрипт запускается из корня проекта или sys.path настроен верно.")
    exit(1)

# 🆕 НОВЫЕ ФУНКЦИИ ДЛЯ EM И EX
def normalize_query(q):
    """Нормализация SQL-запроса для Exact Match: убираем лишние пробелы, приводим к верхнему регистру"""
    if not q: return ""
    return re.sub(r'\s+', ' ', q.strip().upper())

def compute_em(ref_q, gen_q):
    """Exact Match (EM): 1 если запросы идентичны после нормализации, иначе 0"""
    return int(normalize_query(ref_q) == normalize_query(gen_q))

def compute_ex(ref_data, gen_data):
    """Execution Accuracy (EX): 1 если таблицы результатов идентичны (порядок строк не важен), иначе 0"""
    if ref_data is None or gen_data is None:
        return 0
    if not ref_data and not gen_data:
        return 1
    if len(ref_data) != len(gen_data):
        return 0
    
    # Проверяем идентичность столбцов
    ref_cols = sorted(ref_data[0].keys())
    gen_cols = sorted(gen_data[0].keys())
    if ref_cols != gen_cols:
        return 0
    
    # Приводим строки к сортируемому виду (игнорируем порядок строк в ответе БД)
    def to_sortable(table):
        return sorted([tuple(sorted((k, str(v)) for k, v in r.items())) for r in table])
    
    return int(to_sortable(ref_data) == to_sortable(gen_data))

# ⚙️ КОНФИГУРАЦИЯ
TEST_CASES_PATH = "./tests/data/test_cases.json"
RESULTS_PATH    = "./tests/data/metrics_results.json"
EMBED_MODEL     = "sergeyzh/BERTA"  # Модель из статьи. При первом запуске скачается (~1.3 ГБ)
W_WEIGHT        = 0.5                            # Баланс между SC и ST (согласно статье)

def load_embedding_model():
    print(f"📥 Загрузка модели эмбеддингов: {EMBED_MODEL}...")
    try:
        return SentenceTransformer(EMBED_MODEL)
    except Exception:
        print("Модель не найдена, используется лёгкий fallback: all-MiniLM-L6-v2")
        return SentenceTransformer("all-MiniLM-L6-v2")

def compute_sc(model, q1, q2):
    """Семантическая схожесть (Cosine Similarity над эмбеддингами кода)"""
    emb1 = model.encode(q1, normalize_embeddings=True)
    emb2 = model.encode(q2, normalize_embeddings=True)
    return float(np.dot(emb1, emb2))

def get_columns_from_table(data):
    """Преобразует результат запроса в dict {имя_столбца: [значения]} и кол-во строк"""
    if not data:
        return {}, 0
    cols = list(data[0].keys())
    col_data = {c: [str(row[c]) for row in data] for c in cols}
    return col_data, len(data)

def compute_st(gen_data, ref_data):
    """Схожесть таблиц результатов по формуле из статьи"""
    if not gen_data or not ref_data:
        return 0.0

    gen_cols, gen_rows = get_columns_from_table(gen_data)
    ref_cols, ref_rows = get_columns_from_table(ref_data)
    max_rows = max(gen_rows, ref_rows)
    if max_rows == 0:
        return 0.0

    n, m = len(gen_cols), len(ref_cols)
    if n == 0 or m == 0:
        return 0.0

    min_dists = []
    # Для каждого столбца сгенерированного запроса ищем ближайший в эталоне
    for gc_vals in gen_cols.values():
        gc_str = "|".join(gc_vals)  # Безопасное склеивание для сравнения
        best_dist = float('inf')
        for rc_vals in ref_cols.values():
            rc_str = "|".join(rc_vals)
            d = edit_distance(gc_str, rc_str)
            if d < best_dist:
                best_dist = d
        # Нормализация по количеству строк (формула статьи)
        min_dists.append(best_dist / max_rows)
    st = 1.0 - (1.0 / max(n, m)) * sum(min_dists)
    return max(0.0, min(1.0, st))

def compute_qas(sc, st, w=W_WEIGHT):
    """Query Affinity Score: взвешенная сумма"""
    return (1 - w) * sc + w * st

def safe_execute(query):
    """Безопасный вызов executeQuery с обработкой ошибок платформы"""
    try:
        result = executeQuery(query)
        if isinstance(result, dict) and "data" in result:
            return result["data"]
        return []
    except Exception as e:
        print(f"Ошибка выполнения запроса: {e}")
        return None

def main():
    model = load_embedding_model()
    
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)
    
    # Очистка пробелов в ключах JSON (особенность вашего файла)
    test_cases = []
    for tc in raw_cases.get("test_cases", []):
        test_cases.append({k.strip(): v for k, v in tc.items()})

    results = []
    print(f"Начало обработки {len(test_cases)} тест-кейсов...")

    for i, tc in enumerate(test_cases, 1):
        tc_id = tc["id"]
        ref_q = tc["reference_query"]
        gen_q = tc["generate_query"]

        print(f"[{i}/{len(test_cases)}] {tc_id}...")
        
        em = compute_em(ref_q, gen_q)

        # 1. SC: Семантическая схожесть запросов
        sc = compute_sc(model, ref_q, gen_q)

        # 2. ST: Схожесть таблиц результатов
        ref_data = safe_execute(ref_q)
        gen_data = safe_execute(gen_q)

        ex = compute_ex(ref_data, gen_data)

        if ref_data is None or gen_data is None:
            st = 0.0
        elif ref_q.strip() == gen_q.strip():
            st = 1.0
        else:
            st = compute_st(gen_data, ref_data)

        # 3. QAS: Итоговый скор
        qas = compute_qas(sc, st)

        results.append({
            "id": tc_id,
            "user_question": tc.get("user_question", ""),
            "reference_query": ref_q,
            "generated_query": gen_q,
            "EM": em,
            "EX": ex,
            "SC": round(sc, 2),
            "ST": round(st, 2),
            "QAS": round(qas, 2),
            "ref_exec_ok": ref_data is not None,
            "gen_exec_ok": gen_data is not None
        })

    # Сохранение
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Вывод статистики
    print("\nОбработка завершена!")
    print(f"Результаты сохранены: {RESULTS_PATH}")
    print(f"Средние значения:")
    print(f"EM (Exact Match): {np.mean([r['EM'] for r in results]):.2f}")
    print(f"EX (Execution Accuracy): {np.mean([r['EX'] for r in results]):.2f}")
    print(f"SC (Semantic): {np.mean([r['SC'] for r in results]):.2f}")
    print(f"ST (Table):    {np.mean([r['ST'] for r in results]):.2f}")
    print(f"QAS (Affinity):{np.mean([r['QAS'] for r in results]):.2f}")
    
    # Топ-3 лучших и худших по QAS
    sorted_res = sorted(results, key=lambda x: x["QAS"], reverse=True)
    print(f"\nТоп-3 лучших (QAS): {', '.join([r['id'] for r in sorted_res[:3]])}")
    print(f"Топ-3 худших (QAS): {', '.join([r['id'] for r in sorted_res[-3:]])}")

if __name__ == "__main__":
    main()