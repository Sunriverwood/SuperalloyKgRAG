# Copyright 2025 SUNRIVERWOOD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
