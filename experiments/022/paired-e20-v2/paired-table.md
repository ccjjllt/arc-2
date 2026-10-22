| 政策或诊断 | 正确 query / 27 | 全部query正确 task / 20 | 相对4B 新增 / 丢失 / 净增 | 额外逻辑27B task调用 |
| --- | --- | --- | --- | --- |
| B4 | 3 / 27 | 2 / 20 | — | 0 |
| Q27 | 2 / 27 | 2 / 20 | 1 / 2 / -1 | 20 |
| FillMissing | 3 / 27 | 2 / 20 | 0 / 0 / +0 | 5 |
| ReplaceSecond | 4 / 27 | 3 / 20 | 1 / 0 / +1 | 20 |
| 4B ScoredCandidateOracle | 5 / 27 | — | 仅诊断上界 | 不代表可提交政策 |
| 4B已选两槽与27B oracle并集 | 4 / 27 | — | 仅诊断上界 | 不代表可提交政策 |
| 4B评分候选与27B oracle并集 | 6 / 27 | — | 仅诊断上界 | 不代表可提交政策 |
| 4B RawCandidateOracle | 5 / 27 | — | 仅所给raw导出范围 | 不代表可提交政策 |
| 4B原始导出候选与27B oracle并集 | 6 / 27 | — | 仅所给raw导出范围 | 不代表可提交政策 |

所有query保留。25项沿用主官方revision、2项沿用完整输入匹配的旧官方revision；这是公开E20配对诊断。
Q27为同一Direct1结果的重复双槽。两种融合规则已预声明，不按答案选择题或阈值；未新增推理，也未验证12小时全量融合。
ScoredCandidateOracle仅含已评分排名池；RawCandidateOracle只有提供原始合法逆变换grid池时才计算，缺失记未确认，不能据此归因训练不足。
实际完整E20成本和所需逻辑调用见 cost-diagnosis.json；请求时长之和不能当阶段墙钟。
