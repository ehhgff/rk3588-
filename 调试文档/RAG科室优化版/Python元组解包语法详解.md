# Python 元组解包语法详解

## 一、什么是元组解包

**元组解包（Tuple Unpacking）**是 Python 的一种语法特性，允许将一个可迭代对象（元组、列表等）的元素一次性赋值给多个变量。

```python
# 基本语法
a, b, c = (1, 2, 3)
# 结果: a = 1, b = 2, c = 3
```

---

## 二、基本示例

### 2.1 简单解包
```python
# 元组解包
coordinates = (10, 20)
x, y = coordinates
print(x)  # 10
print(y)  # 20

# 等价于
x = coordinates[0]  # 10
y = coordinates[1]  # 20
```

### 2.2 列表解包
```python
# 列表也可以解包
colors = ['red', 'green', 'blue']
r, g, b = colors
print(r)  # 'red'
print(g)  # 'green'
print(b)  # 'blue'
```

### 2.3 字符串解包
```python
# 字符串也是可迭代对象
letters = 'abc'
a, b, c = letters
print(a)  # 'a'
print(b)  # 'b'
print(c)  # 'c'
```

---

## 三、在字典遍历中的应用

### 3.1 实际代码示例
```python
DEPT_KEYWORDS = {
    '儿科': ['孩子', '发烧'],
    '妇产科': ['怀孕', '月经'],
}

for dept, keywords in DEPT_KEYWORDS.items():
    print(f"{dept}: {keywords}")
```

### 3.2 分步拆解

**Step 1: `.items()` 返回什么**
```python
items = DEPT_KEYWORDS.items()
print(list(items))
# [('儿科', ['孩子', '发烧']), ('妇产科', ['怀孕', '月经'])]
```

**Step 2: 每次迭代得到一个元组**
```python
# 第1次迭代
item = ('儿科', ['孩子', '发烧'])

# 第2次迭代
item = ('妇产科', ['怀孕', '月经'])
```

**Step 3: 元组解包**
```python
# 第1次迭代
dept, keywords = ('儿科', ['孩子', '发烧'])
# dept = '儿科'
# keywords = ['孩子', '发烧']

# 第2次迭代
dept, keywords = ('妇产科', ['怀孕', '月经'])
# dept = '妇产科'
# keywords = ['怀孕', '月经']
```

### 3.3 解包过程图解

```
DEPT_KEYWORDS.items()
         ↓
┌─────────────────────────────────────────┐
│  ('儿科', ['孩子', '发烧'])              │
│  ('妇产科', ['怀孕', '月经'])            │
└─────────────────────────────────────────┘
         ↓
    第1次迭代
         ↓
┌─────────────────┐     ┌───────────────┐
│   ('儿科',      │     │  ['孩子',     │
│    ['孩子',     │  →  │   '发烧']     │
│    '发烧'])     │     │               │
└─────────────────┘     └───────────────┘
         ↓                    ↓
      dept='儿科'      keywords=['孩子', '发烧']
```

---

## 四、高级解包技巧

### 4.1 使用 `*` 捕获剩余元素
```python
# 捕获剩余元素
first, *rest = [1, 2, 3, 4, 5]
print(first)  # 1
print(rest)   # [2, 3, 4, 5]

# 中间捕获
first, *middle, last = [1, 2, 3, 4, 5]
print(first)   # 1
print(middle)  # [2, 3, 4]
print(last)    # 5
```

### 4.2 忽略不需要的元素
```python
# 使用 _ 忽略不需要的元素
dept, _ = ('儿科', ['孩子', '发烧'])  # 只需要科室名，忽略关键词
print(dept)  # '儿科'

# 忽略多个元素
first, _, third, _ = (1, 2, 3, 4)
print(first)   # 1
print(third)   # 3
```

### 4.3 嵌套解包
```python
# 嵌套元组解包
data = ('儿科', (100, ['孩子', '发烧']))
dept, (count, keywords) = data
print(dept)      # '儿科'
print(count)     # 100
print(keywords)  # ['孩子', '发烧']
```

---

## 五、实际应用场景

### 5.1 交换变量
```python
# 传统方式
temp = a
a = b
b = temp

# 使用解包（Pythonic）
a, b = b, a
```

### 5.2 函数返回多个值
```python
def get_user():
    return '张三', 25, '北京'  # 返回元组

name, age, city = get_user()
print(f"{name}, {age}岁, 来自{city}")
# 输出: 张三, 25岁, 来自北京
```

### 5.3 遍历字典
```python
# 遍历字典键值对
for key, value in my_dict.items():
    print(f"{key}: {value}")

# 遍历 enumerate
for index, item in enumerate(['a', 'b', 'c']):
    print(f"{index}: {item}")
# 0: a
# 1: b
# 2: c

# 遍历 zip
names = ['Alice', 'Bob']
ages = [25, 30]
for name, age in zip(names, ages):
    print(f"{name} is {age}")
# Alice is 25
# Bob is 30
```

---

## 六、常见错误

### 6.1 变量数量不匹配
```python
# 错误：变量数量少于元素数量
a, b = (1, 2, 3)  # ValueError: too many values to unpack

# 错误：变量数量多于元素数量
a, b, c = (1, 2)  # ValueError: not enough values to unpack
```

### 6.2 解包非可迭代对象
```python
# 错误：解包非可迭代对象
a, b = 10  # TypeError: cannot unpack non-iterable int object
```

### 6.3 解包空对象
```python
# 错误：解包空元组
a, b = ()  # ValueError: not enough values to unpack
```

---

## 七、在 RAG 科室优化中的应用

### 7.1 完整代码示例
```python
DEPT_KEYWORDS = {
    '儿科': ['孩子', '儿童', '婴儿', '宝宝', '小儿', '发烧', '咳嗽', '疫苗'],
    '妇产科': ['怀孕', '孕妇', '分娩', '月经', '妇科', '产科', '备孕', '流产'],
    '皮肤科': ['皮肤', '痘痘', '湿疹', '过敏', '瘙痒', '皮疹', '痤疮', '红斑'],
}

def detect_departments(query):
    """检测查询可能所属的科室"""
    matched = []
    for dept, keywords in DEPT_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            matched.append(dept)
    return matched

# 测试
result = detect_departments("孩子发烧了")
print(result)  # ['儿科']
```

### 7.2 执行流程
```
查询: '孩子发烧了'

第1轮迭代:
    dept = '儿科'
    keywords = ['孩子', '儿童', '婴儿', '宝宝', '小儿', '发烧', '咳嗽', '疫苗']
    any() 检查:
        '孩子' in '孩子发烧了'? → True ✓
    匹配成功，添加 '儿科'

第2轮迭代:
    dept = '妇产科'
    keywords = ['怀孕', '孕妇', '分娩', '月经', '妇科', '产科', '备孕', '流产']
    any() 检查:
        '怀孕' in '孩子发烧了'? → False
        '孕妇' in '孩子发烧了'? → False
        ... 全部不匹配
    不匹配，跳过

第3轮迭代:
    dept = '皮肤科'
    ... 全部不匹配

最终结果: ['儿科']
```

---

## 八、总结

| 特性 | 说明 |
|------|------|
| **语法** | `a, b, c = iterable` |
| **原理** | 将可迭代对象的元素依次赋值给变量 |
| **要求** | 变量数量必须与元素数量匹配 |
| **优势** | 代码简洁、可读性强、Pythonic |

### 核心要点
1. 元组解包可以同时获取多个值
2. 适用于遍历字典、列表等可迭代对象
3. 可以使用 `*` 捕获剩余元素
4. 使用 `_` 忽略不需要的元素
5. 变量数量必须与元素数量匹配

---

*文档版本: v1.0*
*最后更新: 2024-04-06*
