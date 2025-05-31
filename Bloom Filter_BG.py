# Probabilistic Data Structures
# Анализ на Bloom Filter

import numpy as np
import matplotlib.pyplot as plt
import hashlib
from collections import defaultdict

# ============================================================================
# ТЕОРЕТИЧНА ЧАСТ
# ============================================================================

print("=== ВЕРОЯТНОСТНИ СТРУКТУРИ ОТ ДАННИ ===")
print()

print("1. Какво е структура от данни?")
print("   Структура от данни е начин за организиране и съхраняване на данни")
print("   в компютъра, който позволява ефективен достъп и модификация.")
print()

print("2. Какво е вероятностна структура от данни?")
print("   Това е структура, която използва рандомизация в своята работа")
print("   и може да даде неточни отговори с определена вероятност.")
print("   Предимства:")
print("   - Много по-малко използване на памет")
print("   - По-бързи операции")
print("   - Скалируемост за големи данни")
print()

print("3. BLOOM FILTER")
print("   Bloom filter е вероятностна структура за проверка на принадлежност")
print("   към множество. Може да каже:")
print("   - 'Определено НЕ е в множеството' (100% точност)")
print("   - 'Възможно е да е в множеството' (може да греши)")
print()

# ============================================================================
# ИМПЛЕМЕНТАЦИЯ НА BLOOM FILTER
# ============================================================================

class BloomFilter:
    """
    Проста имплементация на Bloom Filter
    """
    
    def __init__(self, size, hash_count):
        """
        size: размер на битовия масив
        hash_count: брой хеш функции
        """
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [0] * size
        self.items_added = 0
    
    def _hash(self, item, seed):
        """Хеш функция с различни seeds"""
        hash_obj = hashlib.md5((str(item) + str(seed)).encode())
        return int(hash_obj.hexdigest(), 16) % self.size
    
    def add(self, item):
        """Добавяне на елемент"""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = 1
        self.items_added += 1
    
    def check(self, item):
        """Проверка дали елементът е в множеството"""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if self.bit_array[index] == 0:
                return False  # Определено НЕ е в множеството
        return True  # Възможно е да е в множеството
    
    def false_positive_rate(self):
        """Теоретична вероятност за false positive"""
        # Формула: (1 - e^(-k*n/m))^k
        # k = hash_count, n = items_added, m = size
        if self.items_added == 0:
            return 0
        
        k = self.hash_count
        n = self.items_added
        m = self.size
        
        return (1 - np.exp(-k * n / m)) ** k

# ============================================================================
# ДЕМОНСТРАЦИЯ И АНАЛИЗ
# ============================================================================

print("=== ДЕМОНСТРАЦИЯ НА BLOOM FILTER ===")
print()

# Създаване на Bloom filter
bf = BloomFilter(size=1000, hash_count=3)

# Добавяне на думи
words_in_set = ["apple", "banana", "cherry", "date", "elderberry"]
for word in words_in_set:
    bf.add(word)

print(f"Добавени думи: {words_in_set}")
print()

# Тестване
test_words = ["apple", "grape", "banana", "kiwi", "cherry", "mango"]
print("Резултати от проверки:")
for word in test_words:
    result = bf.check(word)
    actual = word in words_in_set
    status = "✓" if result == actual else "✗ FALSE POSITIVE"
    print(f"  {word:12} -> {result:5} (реално: {actual:5}) {status}")

print(f"\nТеоретична вероятност за false positive: {bf.false_positive_rate():.4f}")

# ============================================================================
# АНАЛИЗ НА ПРОИЗВОДИТЕЛНОСТТА
# ============================================================================

print("\n=== АНАЛИЗ НА ПРОИЗВОДИТЕЛНОСТТА ===")
print()

# Сравнение с обикновен set
import time

# Тест с голям брой елементи
n_items = 100000
test_items = [f"item_{i}" for i in range(n_items)]
test_queries = [f"item_{i}" for i in range(0, n_items, 100)]  # Всеки 100-ти

# Python set
start_time = time.time()
python_set = set(test_items)
set_creation_time = time.time() - start_time

start_time = time.time()
for item in test_queries:
    item in python_set
set_query_time = time.time() - start_time

# Bloom filter
bf_large = BloomFilter(size=n_items//2, hash_count=5)  # По-малък размер

start_time = time.time()
for item in test_items:
    bf_large.add(item)
bf_creation_time = time.time() - start_time

start_time = time.time()
for item in test_queries:
    bf_large.check(item)
bf_query_time = time.time() - start_time

print("Сравнение Python set vs Bloom Filter:")
print(f"Брой елементи: {n_items:,}")
print(f"Брой заявки: {len(test_queries):,}")
print()
print("Време за създаване:")
print(f"  Python set:   {set_creation_time:.4f}s")
print(f"  Bloom filter: {bf_creation_time:.4f}s")
print()
print("Време за заявки:")
print(f"  Python set:   {set_query_time:.4f}s")
print(f"  Bloom filter: {bf_query_time:.4f}s")
print()

# Използване на памет
import sys
set_memory = sys.getsizeof(python_set) + sum(sys.getsizeof(item) for item in python_set)
bf_memory = sys.getsizeof(bf_large.bit_array)

print("Използване на памет:")
print(f"  Python set:   {set_memory:,} bytes")
print(f"  Bloom filter: {bf_memory:,} bytes")
print(f"  Спестяване:   {(1 - bf_memory/set_memory)*100:.1f}%")
print()

# ============================================================================
# FALSE POSITIVE АНАЛИЗ
# ============================================================================

print("=== АНАЛИЗ НА FALSE POSITIVE RATE ===")

# Тестване с различни параметри
sizes = [500, 1000, 2000, 5000]
hash_counts = [1, 3, 5, 7]
n_test_items = 1000

results = []

for size in sizes:
    for hash_count in hash_counts:
        bf_test = BloomFilter(size, hash_count)
        
        # Добавяме първата половина от елементите
        for i in range(n_test_items // 2):
            bf_test.add(f"item_{i}")
        
        # Тестваме втората половина (които НЕ са добавени)
        false_positives = 0
        for i in range(n_test_items // 2, n_test_items):
            if bf_test.check(f"item_{i}"):
                false_positives += 1
        
        actual_fp_rate = false_positives / (n_test_items // 2)
        theoretical_fp_rate = bf_test.false_positive_rate()
        
        results.append({
            'size': size,
            'hash_count': hash_count,
            'actual_fp': actual_fp_rate,
            'theoretical_fp': theoretical_fp_rate
        })

# Визуализация на резултатите
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# График 1: Влияние на размера
for hc in hash_counts:
    actual_rates = [r['actual_fp'] for r in results if r['hash_count'] == hc]
    theoretical_rates = [r['theoretical_fp'] for r in results if r['hash_count'] == hc]
    ax1.plot(sizes, actual_rates, 'o-', label=f'Реален (k={hc})')
    ax1.plot(sizes, theoretical_rates, '--', label=f'Теоретичен (k={hc})')

ax1.set_xlabel('Размер на Bloom Filter')
ax1.set_ylabel('False Positive Rate')
ax1.set_title('Влияние на размера върху FP Rate')
ax1.legend()
ax1.grid(True)

# График 2: Влияние на броя хеш функции
for size in [1000, 5000]:
    actual_rates = [r['actual_fp'] for r in results if r['size'] == size]
    theoretical_rates = [r['theoretical_fp'] for r in results if r['size'] == size]
    ax2.plot(hash_counts, actual_rates, 'o-', label=f'Реален (m={size})')
    ax2.plot(hash_counts, theoretical_rates, '--', label=f'Теоретичен (m={size})')

ax2.set_xlabel('Брой хеш функции')
ax2.set_ylabel('False Positive Rate')
ax2.set_title('Влияние на броя хеш функции върху FP Rate')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# ============================================================================
# ПРАКТИЧЕСКО ПРИЛОЖЕНИЕ: SPELL CHECKER
# ============================================================================

print("\n=== ПРАКТИЧЕСКО ПРИЛОЖЕНИЕ: SPELL CHECKER ===")
print()

class SpellChecker:
    """
    Прост spell checker използващ Bloom Filter
    """
    
    def __init__(self, dictionary_words, filter_size=None, hash_count=5):
        self.dictionary = set(dictionary_words)  # За точни проверки
        
        # Bloom filter за бърза първична проверка
        if filter_size is None:
            filter_size = len(dictionary_words) * 2
        
        self.bloom_filter = BloomFilter(filter_size, hash_count)
        
        # Добавяме всички думи в Bloom filter
        for word in dictionary_words:
            self.bloom_filter.add(word.lower())
    
    def is_correct_bloom_only(self, word):
        """Проверка само с Bloom filter (може да има false positives)"""
        return self.bloom_filter.check(word.lower())
    
    def is_correct_exact(self, word):
        """Точна проверка с dictionary"""
        return word.lower() in self.dictionary
    
    def is_correct_hybrid(self, word):
        """Хибридна проверка: първо Bloom filter, после exact"""
        if not self.bloom_filter.check(word.lower()):
            return False  # Определено грешна дума
        return word.lower() in self.dictionary  # Точна проверка

# Демонстрация на spell checker
dictionary = [
    "apple", "application", "apply", "approach", "appropriate",
    "banana", "beautiful", "become", "because", "before",
    "computer", "science", "python", "programming", "algorithm",
    "data", "structure", "analysis", "statistics", "probability"
]

spell_checker = SpellChecker(dictionary)

test_words = ["apple", "aple", "computer", "computor", "science", "sciance", "xyz"]

print("Тест на spell checker:")
print("Дума       | Bloom | Exact | Hybrid | Статус")
print("-" * 50)

for word in test_words:
    bloom_result = spell_checker.is_correct_bloom_only(word)
    exact_result = spell_checker.is_correct_exact(word)
    hybrid_result = spell_checker.is_correct_hybrid(word)
    
    if bloom_result and not exact_result:
        status = "FALSE POSITIVE"
    elif exact_result:
        status = "CORRECT"
    else:
        status = "INCORRECT"
    
    print(f"{word:10} | {bloom_result:5} | {exact_result:5} | {hybrid_result:6} | {status}")

print(f"\nBloom Filter статистики:")
print(f"Размер: {spell_checker.bloom_filter.size}")
print(f"Хеш функции: {spell_checker.bloom_filter.hash_count}")
print(f"Думи в речника: {len(dictionary)}")
print(f"Теоретична FP rate: {spell_checker.bloom_filter.false_positive_rate():.4f}")

# ============================================================================
# ЗАКЛЮЧЕНИЕ
# ============================================================================

print("\n=== ЗАКЛЮЧЕНИЕ ===")
print()
print("Bloom Filter предимства:")
print("- Много малко използване на памет (битов масив)")
print("- Бързи операции O(k) където k е броят хеш функции") 
print("- Никога няма false negatives (ако каже 'НЕ', то е 100% вярно)")
print("- Скалируем за много големи данни")
print()
print("Bloom Filter недостатъци:")
print("- Може да има false positives")
print("- Не може да се премахват елементи (има варианти като Counting Bloom Filter)")
print("- Не съхранява самите данни, само проверява принадлежност")
print()
print("Приложения:")
print("- Spell checkers (бърза проверка преди скъпа операция)")
print("- Web caching (проверка дали URL е в кеша)")
print("- Database query optimization")
print("- Network security (blacklist проверки)")
print("- Big Data приложения (например в Apache Spark)")
