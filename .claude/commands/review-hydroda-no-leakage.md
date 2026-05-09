# /review-hydroda-no-leakage

对当前代码或实验结果做 no-leakage 审查。

要求读取：

```text
CLAUDE.md
checklists/no_leakage_checklist.md
context/02_DATA_AND_LEAKAGE_CONTRACT.md
context/04_KDATE_SPLIT_PROTOCOL.md
```

检查：

```text
split leakage
normalization leakage
region leakage
training leakage
evaluation leakage
```

输出：

```text
结论：pass / warning / fail
发现：
证据文件：
需要修改：
阻塞级别：
```
