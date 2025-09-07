#!/usr/bin/env python3
"""
测试 SQL 抽取函数的脚本
"""

import re

def extract_sql_from_output(output_text):
    """
    从模型输出中提取 SQL 查询
    支持多种格式的 SQL 代码块
    """
    # 1. 优先提取 Qwen XML 格式的 <sql> 标签
    xml_sql_pattern = r'<sql>\s*(.*?)\s*</sql>'
    match = re.search(xml_sql_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 2. 尝试匹配 <answer>```sql ... ``` </answer> 格式
    answer_pattern = r'<answer>\s*```sql\s*(.*?)\s*```\s*</answer>'
    match = re.search(answer_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 3. 尝试匹配普通的 ```sql ... ``` 格式
    sql_pattern = r'```sql\s*(.*?)\s*```'
    match = re.search(sql_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 4. 尝试匹配 ``` ... ``` 格式（无 sql 标识）
    code_pattern = r'```\s*(.*?)\s*```'
    match = re.search(code_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        sql_candidate = match.group(1).strip()
        # 简单检查是否像 SQL（包含 SELECT、INSERT、UPDATE、DELETE 等关键词）
        if any(keyword in sql_candidate.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
            return sql_candidate

    # 5. 直接提取首个 SQL 语句 - 改进版本
    # 匹配从 SELECT/INSERT/UPDATE/DELETE 开始到分号或行末的完整语句
    sql_line_pattern = r'((?:SELECT|INSERT|UPDATE|DELETE)(?:[^;])+(?:;|$))'
    match = re.search(sql_line_pattern, output_text, re.IGNORECASE | re.MULTILINE)
    if match:
        sql_candidate = match.group(1).strip()
        # 确保不是只有一个关键词
        if len(sql_candidate.split()) > 1:
            return sql_candidate
    
    # 6. 更宽松的匹配 - 匹配多行 SQL
    multiline_sql_pattern = r'((?:SELECT|INSERT|UPDATE|DELETE)(?:.|\n)*?)(?:\n\s*\n|\n\s*$|$)'
    match = re.search(multiline_sql_pattern, output_text, re.IGNORECASE)
    if match:
        sql_candidate = match.group(1).strip()
        # 确保有基本的 SQL 结构
        if len(sql_candidate.split()) > 2 and ('FROM' in sql_candidate.upper() or 'INTO' in sql_candidate.upper()):
            return sql_candidate

    # 7. 按行查找最长的 SQL 语句
    longest_sql = None
    max_length = 0
    for line in output_text.splitlines():
        line = line.strip()
        if line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE')):
            if len(line) > max_length:
                longest_sql = line
                max_length = len(line)
    
    if longest_sql and len(longest_sql.split()) > 2:
        return longest_sql
    
    return None

# 测试用例
test_cases = [
    # XML 格式
    """<think>Let me analyze this step by step</think>
<sql>SELECT * FROM users WHERE age > 25</sql>""",
    
    # 代码块格式
    """```sql
SELECT name, email FROM customers WHERE city = 'New York'
```""",
    
    # 普通文本中的 SQL
    """The SQL query would be:
SELECT COUNT(*) FROM orders WHERE status = 'completed'
This will give us the total count.""",
    
    # 多行 SQL
    """Here's the query:
SELECT u.name, o.total 
FROM users u 
JOIN orders o ON u.id = o.user_id 
WHERE o.date > '2023-01-01'""",
    
    # 只有 SELECT 的情况（应该被过滤）
    """The answer is SELECT""",
]

if __name__ == "__main__":
    for i, test in enumerate(test_cases):
        result = extract_sql_from_output(test)
        print(f"测试 {i+1}:")
        print(f"输入: {test[:50]}...")
        print(f"提取结果: {result}")
        print("-" * 50)
