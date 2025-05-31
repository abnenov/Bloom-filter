# Probabilistic Data Structures
# Bloom Filter Analysis

import numpy as np
import matplotlib.pyplot as plt
import hashlib
from collections import defaultdict

# ============================================================================
# THEORETICAL PART
# ============================================================================

print("=== PROBABILISTIC DATA STRUCTURES ===")
print()

print("1. What is a data structure?")
print("   A data structure is a way of organizing and storing data")
print("   in a computer that allows efficient access and modification.")
print()

print("2. What is a probabilistic data structure?")
print("   This is a structure that uses randomization in its operation")
print("   and may give inaccurate answers with a certain probability.")
print("   Advantages:")
print("   - Much lower memory usage")
print("   - Faster operations")
print("   - Scalability for large data")
print()

print("3. BLOOM FILTER")
print("   A Bloom filter is a probabilistic structure for membership testing")
print("   in a set. It can say:")
print("   - 'Definitely NOT in the set' (100% accuracy)")
print("   - 'Possibly in the set' (may be wrong)")
print()

# ============================================================================
# BLOOM FILTER IMPLEMENTATION
# ============================================================================

class BloomFilter:
    """
    Simple implementation of a Bloom Filter
    """
    
    def __init__(self, size, hash_count):
        """
        size: size of the bit array
        hash_count: number of hash functions
        """
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [0] * size
        self.items_added = 0
    
    def _hash(self, item, seed):
        """Hash function with different seeds"""
        hash_obj = hashlib.md5((str(item) + str(seed)).encode())
        return int(hash_obj.hexdigest(), 16) % self.size
    
    def add(self, item):
        """Add an element"""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = 1
        self.items_added += 1
    
    def check(self, item):
        """Check if the element is in the set"""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if self.bit_array[index] == 0:
                return False  # Definitely NOT in the set
        return True  # Possibly in the set
    
    def false_positive_rate(self):
        """Theoretical probability for false positive"""
        # Formula: (1 - e^(-k*n/m))^k
        # k = hash_count, n = items_added, m = size
        if self.items_added == 0:
            return 0
        
        k = self.hash_count
        n = self.items_added
        m = self.size
        
        return (1 - np.exp(-k * n / m)) ** k

# ============================================================================
# DEMONSTRATION AND ANALYSIS
# ============================================================================

print("=== BLOOM FILTER DEMONSTRATION ===")
print()

# Create a Bloom filter
bf = BloomFilter(size=1000, hash_count=3)

# Add words
words_in_set = ["apple", "banana", "cherry", "date", "elderberry"]
for word in words_in_set:
    bf.add(word)

print(f"Added words: {words_in_set}")
print()

# Testing
test_words = ["apple", "grape", "banana", "kiwi", "cherry", "mango"]
print("Check results:")
for word in test_words:
    result = bf.check(word)
    actual = word in words_in_set
    status = "✓" if result == actual else "✗ FALSE POSITIVE"
    print(f"  {word:12} -> {result:5} (actual: {actual:5}) {status}")

print(f"\nTheoretical false positive probability: {bf.false_positive_rate():.4f}")

# ============================================================================
# PERFORMANCE ANALYSIS
# ============================================================================

print("\n=== PERFORMANCE ANALYSIS ===")
print()

# Comparison with regular set
import time

# Test with large number of elements
n_items = 100000
test_items = [f"item_{i}" for i in range(n_items)]
test_queries = [f"item_{i}" for i in range(0, n_items, 100)]  # Every 100th

# Python set
start_time = time.time()
python_set = set(test_items)
set_creation_time = time.time() - start_time

start_time = time.time()
for item in test_queries:
    item in python_set
set_query_time = time.time() - start_time

# Bloom filter
bf_large = BloomFilter(size=n_items//2, hash_count=5)  # Smaller size

start_time = time.time()
for item in test_items:
    bf_large.add(item)
bf_creation_time = time.time() - start_time

start_time = time.time()
for item in test_queries:
    bf_large.check(item)
bf_query_time = time.time() - start_time

print("Comparison Python set vs Bloom Filter:")
print(f"Number of elements: {n_items:,}")
print(f"Number of queries: {len(test_queries):,}")
print()
print("Creation time:")
print(f"  Python set:   {set_creation_time:.4f}s")
print(f"  Bloom filter: {bf_creation_time:.4f}s")
print()
print("Query time:")
print(f"  Python set:   {set_query_time:.4f}s")
print(f"  Bloom filter: {bf_query_time:.4f}s")
print()

# Memory usage
import sys
set_memory = sys.getsizeof(python_set) + sum(sys.getsizeof(item) for item in python_set)
bf_memory = sys.getsizeof(bf_large.bit_array)

print("Memory usage:")
print(f"  Python set:   {set_memory:,} bytes")
print(f"  Bloom filter: {bf_memory:,} bytes")
print(f"  Memory saved: {(1 - bf_memory/set_memory)*100:.1f}%")
print()

# ============================================================================
# FALSE POSITIVE ANALYSIS
# ============================================================================

print("=== FALSE POSITIVE RATE ANALYSIS ===")

# Testing with different parameters
sizes = [500, 1000, 2000, 5000]
hash_counts = [1, 3, 5, 7]
n_test_items = 1000

results = []

for size in sizes:
    for hash_count in hash_counts:
        bf_test = BloomFilter(size, hash_count)
        
        # Add first half of elements
        for i in range(n_test_items // 2):
            bf_test.add(f"item_{i}")
        
        # Test second half (which were NOT added)
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

# Visualization of results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Graph 1: Effect of size
for hc in hash_counts:
    actual_rates = [r['actual_fp'] for r in results if r['hash_count'] == hc]
    theoretical_rates = [r['theoretical_fp'] for r in results if r['hash_count'] == hc]
    ax1.plot(sizes, actual_rates, 'o-', label=f'Actual (k={hc})')
    ax1.plot(sizes, theoretical_rates, '--', label=f'Theoretical (k={hc})')

ax1.set_xlabel('Bloom Filter Size')
ax1.set_ylabel('False Positive Rate')
ax1.set_title('Effect of Size on FP Rate')
ax1.legend()
ax1.grid(True)

# Graph 2: Effect of number of hash functions
for size in [1000, 5000]:
    actual_rates = [r['actual_fp'] for r in results if r['size'] == size]
    theoretical_rates = [r['theoretical_fp'] for r in results if r['size'] == size]
    ax2.plot(hash_counts, actual_rates, 'o-', label=f'Actual (m={size})')
    ax2.plot(hash_counts, theoretical_rates, '--', label=f'Theoretical (m={size})')

ax2.set_xlabel('Number of Hash Functions')
ax2.set_ylabel('False Positive Rate')
ax2.set_title('Effect of Hash Functions on FP Rate')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# ============================================================================
# PRACTICAL APPLICATION: SPELL CHECKER
# ============================================================================

print("\n=== PRACTICAL APPLICATION: SPELL CHECKER ===")
print()

class SpellChecker:
    """
    Simple spell checker using Bloom Filter
    """
    
    def __init__(self, dictionary_words, filter_size=None, hash_count=5):
        self.dictionary = set(dictionary_words)  # For exact checks
        
        # Bloom filter for fast initial check
        if filter_size is None:
            filter_size = len(dictionary_words) * 2
        
        self.bloom_filter = BloomFilter(filter_size, hash_count)
        
        # Add all words to Bloom filter
        for word in dictionary_words:
            self.bloom_filter.add(word.lower())
    
    def is_correct_bloom_only(self, word):
        """Check only with Bloom filter (may have false positives)"""
        return self.bloom_filter.check(word.lower())
    
    def is_correct_exact(self, word):
        """Exact check with dictionary"""
        return word.lower() in self.dictionary
    
    def is_correct_hybrid(self, word):
        """Hybrid check: first Bloom filter, then exact"""
        if not self.bloom_filter.check(word.lower()):
            return False  # Definitely wrong word
        return word.lower() in self.dictionary  # Exact check

# Spell checker demonstration
dictionary = [
    "apple", "application", "apply", "approach", "appropriate",
    "banana", "beautiful", "become", "because", "before",
    "computer", "science", "python", "programming", "algorithm",
    "data", "structure", "analysis", "statistics", "probability"
]

spell_checker = SpellChecker(dictionary)

test_words = ["apple", "aple", "computer", "computor", "science", "sciance", "xyz"]

print("Spell checker test:")
print("Word       | Bloom | Exact | Hybrid | Status")
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

print(f"\nBloom Filter statistics:")
print(f"Size: {spell_checker.bloom_filter.size}")
print(f"Hash functions: {spell_checker.bloom_filter.hash_count}")
print(f"Dictionary words: {len(dictionary)}")
print(f"Theoretical FP rate: {spell_checker.bloom_filter.false_positive_rate():.4f}")

# ============================================================================
# COMPLEXITY ANALYSIS
# ============================================================================

print("\n=== COMPLEXITY ANALYSIS ===")
print()

print("Time Complexity:")
print("- Add operation: O(k) where k is number of hash functions")
print("- Check operation: O(k) where k is number of hash functions")
print("- Space complexity: O(m) where m is size of bit array")
print()

print("Comparison with traditional data structures:")
print()
print("| Operation | Hash Set | Bloom Filter | Advantage |")
print("|-----------|----------|--------------|-----------|")
print("| Add       | O(1)*    | O(k)         | Similar   |")
print("| Check     | O(1)*    | O(k)         | Similar   |")
print("| Space     | O(n*w)   | O(m)         | Much less |")
print()
print("* Average case, O(n) worst case")
print("n = number of elements, w = word size, m = bit array size, k = hash functions")
print()

print("Bloom Filter is especially advantageous when:")
print("- Memory is limited")
print("- Network bandwidth is limited")
print("- Exact membership is not critical")
print("- False negatives are not acceptable")
print("- Working with very large datasets")

# ============================================================================
# REAL-WORLD APPLICATIONS
# ============================================================================

print("\n=== REAL-WORLD APPLICATIONS ===")
print()

print("1. Web Browsers:")
print("   - Chrome uses Bloom filters to check malicious URLs")
print("   - Quick check before expensive server lookup")
print()

print("2. Database Systems:")
print("   - Apache Cassandra uses Bloom filters for SSTables")
print("   - Avoids expensive disk reads for non-existent data")
print()

print("3. Content Delivery Networks (CDN):")
print("   - Quick check if content is in cache")
print("   - Reduces cache misses and improves performance")
print()

print("4. Distributed Systems:")
print("   - Apache Spark uses Bloom filters for join optimization")
print("   - Reduces network traffic in distributed joins")
print()

print("5. Cryptocurrency:")
print("   - Bitcoin uses Bloom filters in SPV (Simplified Payment Verification)")
print("   - Allows lightweight clients to verify transactions")

# ============================================================================
# VARIANTS AND EXTENSIONS
# ============================================================================

print("\n=== BLOOM FILTER VARIANTS ===")
print()

print("1. Counting Bloom Filter:")
print("   - Uses counters instead of bits")
print("   - Allows deletion of elements")
print("   - Uses more memory")
print()

print("2. Scalable Bloom Filter:")
print("   - Dynamically adjusts size as more elements are added")
print("   - Maintains target false positive rate")
print()

print("3. Cuckoo Filter:")
print("   - Alternative to Bloom filter")
print("   - Supports deletion")
print("   - Better space efficiency for low false positive rates")
print()

print("4. Quotient Filter:")
print("   - Cache-friendly alternative")
print("   - Supports deletion and merging")
print("   - Better performance on modern hardware")

# ============================================================================
# IMPLEMENTATION OPTIMIZATIONS
# ============================================================================

print("\n=== IMPLEMENTATION OPTIMIZATIONS ===")
print()

class OptimizedBloomFilter:
    """
    More optimized version with better hash functions
    """
    
    def __init__(self, expected_elements, false_positive_rate=0.01):
        """
        Calculate optimal parameters based on requirements
        """
        # Optimal size: m = -(n * ln(p)) / (ln(2)^2)
        self.expected_elements = expected_elements
        self.false_positive_rate = false_positive_rate
        
        ln2 = np.log(2)
        self.size = int(-(expected_elements * np.log(false_positive_rate)) / (ln2 ** 2))
        
        # Optimal number of hash functions: k = (m/n) * ln(2)
        self.hash_count = int((self.size / expected_elements) * ln2)
        
        self.bit_array = [0] * self.size
        self.items_added = 0
        
        print(f"Optimized Bloom Filter created:")
        print(f"  Expected elements: {expected_elements}")
        print(f"  Target FP rate: {false_positive_rate}")
        print(f"  Calculated size: {self.size}")
        print(f"  Hash functions: {self.hash_count}")
    
    def _hash(self, item, seed):
        """Double hashing for better distribution"""
        # Use two different hash functions
        hash1 = hash(str(item)) % self.size
        hash2 = hash(str(item)[::-1]) % self.size
        return (hash1 + seed * hash2) % self.size
    
    def add(self, item):
        """Add element with optimized hashing"""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = 1
        self.items_added += 1
    
    def check(self, item):
        """Check membership with optimized hashing"""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if self.bit_array[index] == 0:
                return False
        return True

# Test optimized version
print("\n=== TESTING OPTIMIZED BLOOM FILTER ===")
optimized_bf = OptimizedBloomFilter(expected_elements=1000, false_positive_rate=0.01)

# Add test elements
test_elements = [f"element_{i}" for i in range(500)]
for element in test_elements:
    optimized_bf.add(element)

# Test false positive rate
false_positives = 0
test_count = 1000
for i in range(500, 500 + test_count):
    if optimized_bf.check(f"element_{i}"):
        false_positives += 1

actual_fp_rate = false_positives / test_count
print(f"\nActual false positive rate: {actual_fp_rate:.4f}")
print(f"Target false positive rate: {optimized_bf.false_positive_rate:.4f}")

# ============================================================================
# CONCLUSION
# ============================================================================

print("\n=== CONCLUSION ===")
print()
print("Bloom Filter Advantages:")
print("- Very low memory usage (bit array)")
print("- Fast operations O(k) where k is number of hash functions") 
print("- Never has false negatives (if it says 'NO', it's 100% correct)")
print("- Scalable for very large data")
print("- Simple to implement and understand")
print()
print("Bloom Filter Disadvantages:")
print("- May have false positives")
print("- Cannot delete elements (variants like Counting BF can)")
print("- Does not store actual data, only checks membership")
print("- Requires careful parameter tuning")
print()
print("Applications:")
print("- Spell checkers (fast check before expensive operation)")
print("- Web caching (check if URL is in cache)")
print("- Database query optimization")
print("- Network security (blacklist checks)")
print("- Big Data applications (e.g., in Apache Spark)")
print("- Distributed systems (reduce network overhead)")
print()
print("When to use Bloom Filters:")
print("- Memory/bandwidth is limited")
print("- False positives are acceptable")
print("- False negatives are NOT acceptable")
print("- Need very fast membership testing")
print("- Working with massive datasets")
