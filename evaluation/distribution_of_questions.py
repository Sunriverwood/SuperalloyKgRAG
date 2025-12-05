import json

# 读取 JSON 文件
with open('../data/evaluation_sets/L4.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取所有 domain 值
domains = set(item['domain'] for item in data)

# 打印结果
print(f"总共有 {len(domains)} 种不同的 domain:")
for domain in sorted(domains):
    print(f"  - {domain}")

# 可选:统计每个 domain 的数量
from collections import Counter
domain_counts = Counter(item['domain'] for item in data)
print("\n各 domain 的数量:")
for domain, count in sorted(domain_counts.items()):
    print(f"  {domain}: {count}")
