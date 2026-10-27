# 032 直接提交最高分公开Notebook进展

用户于2026-10-27明确要求直接使用最高分Notebook提交，不继续031研究。本次CLI按scoreDescending实时列表第一仍为Failed in AIMO；选已核验外部33.89的V1／312063593。

正式预检：CLI submission-limits返回numTotal12、numAllowedNow1；最近10条submission均COMPLETE，最近02729.72，02330.14。原参考Notebook冻结SHA匹配。当前尚未发起正式请求，真实受理与绑定将追加记录。

## 已正式受理

**2026-10-27 16:31:08北京时间，平台接受submission57694986。** 使用官方Kaggle CLI、显式zhuating认证，直接提交 `koushikrudra/failed-in-aimo` **原作者V1**、文件 `submission.json`；没有复制／修改源码、没有改用023 runtime修复副本，也没有加入031或其他研究改动。服务端本次允许直接引用该公开原作者版本，不需要新ordinary GPU会话。

受理response返回ref57694986；CLI随后显示今日remaining0，真实limits为num_today1、num_total13、num_allowed_now0。CLI提示不是评分结果。受理原始记录见 `experiments/032/execution/original-v1-submission-attempt.json`。

随后一次队伍submission列表找到同一ref：submitted_by **zhuating**、team_name **ccjjllt**，文件名及描述正确，URL为 `/code/koushikrudra/failed-in-aimo?scriptVersionId=312063593`，**精确V1／312063593绑定通过**。当前状态 **PENDING**，public／private score均未返回，无error_description；不得把原作者历史33.89填成本队新成绩。真实初始绑定见 `experiments/032/execution/submission-57694986-initial-binding.json`。

原高分Notebook于10月21日精确页面绑定33.89，10月27日本次CLI scoreDescending排序仍第一；最新页面正文未能经web工具读取，未伪称重新逐页面取到分数。最高分排序与精确源版本证据分别保留。

受理及绑定完成后停止远端查询，等待用户完成通知，不自动监控或重复提交。下一次通过官方CLI submissions读57694986真实终态与分数，保持实际scriptVersion312063593绑定。此时本项目已确认正式最好仍02330.14，新submission是否提分尚未确认。
