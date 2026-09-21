# 文物照明保护—展示双目标优化工具

基于 10 路 LED 光谱、18 种文物材料（9 有机颜料 + 3 无机颜料 + 6 基材：新闻纸、
竹纸、宣纸、麻纸、桑皮纸、绢）的保护—展示双目标优化 Web 应用。用户输入文物材料
组成、面积比例、展示时间与画作色调类型，程序以 NSGA-II 搜索保护（最小化整体损伤
D_relic）与展示（最大化效果 F）的 Pareto 解集，并输出**保护优先 / 展示优先 /
综合折中**三类代表照明方案（10 路 LED 权重 + 照度）。

口径来源：《文物照明保护展示双目标优化应用_DeepSeek_Harness完整产品说明_v3》、
《公式集合_初版.docx》、《**基材公式补充.docx**》（正式基材公式）及
《需求确认记录.md》（同目录上级）。

---

## 1. Windows 本地运行步骤

### 1.1 安装 Python

推荐 **Python 3.11**（3.10 亦可）。安装时勾选 *Add python.exe to PATH*。

### 1.2 创建虚拟环境并安装依赖

在项目根目录（本文件所在目录）打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> 说明：`luxpy`（TM-30 备用实现）要求 numpy>=2.0，requirements 已固定；
> 若安装 luxpy 失败，主实现 `colour-science` 仍可单独完成全部颜色指标。

### 1.3 运行自动测试（验收 T1–T13）

```powershell
python -m pytest tests/ -v
```

### 1.4 启动网页应用

```powershell
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。若未自动打开，手动访问该地址。

---

## 2. 界面使用（五区）

| 区域 | 内容 |
|------|------|
| A 数据状态 | 显示 LED/D55 加载状态、波长范围、通道数；可上传文件覆盖默认数据（当前会话生效） |
| B 文物输入 | 勾选材料、输入面积比例 α（总和须为 1，可自动归一化）、展示时间 t、画作色调（冷/暖/中）；勾选基材时以**只读提示**显示其固定 pH（无需输入、不参与计算） |
| C 优化设置 | 高级设置折叠区：种群规模、迭代代数、交叉/变异概率、随机种子 |
| D 运行 | 开始优化按钮、进度条、当前代数 |
| E 结果 | Pareto 前沿、三类代表方案表、代表 SPD 图（含 D55）、10 通道权重图、颜色参数、下载按钮；"计算明细"折叠区可查看各材料 K/f/D_raw/D0/D_norm/α·D_norm |

下载项：Pareto CSV、代表方案 CSV、代表 SPD CSV、Pareto PNG、全部 XLSX。

---

## 3. 数据与模型

- 默认数据：`data/led_10_spd.xlsx`（10 路 LED SPD）、`data/d55_spd.xlsx`（D55 参考光源）。
  读取按位置解析（第 1 列波长 + 后续列 SPD），不依赖表头名称；波长须完整覆盖 380–780 nm。
- **LED 数据为可替换输入**：当前 `data/led_10_spd.xlsx` 为临时模拟数据。正式数据可直接
  替换该文件，或在界面 A 区上传（格式：第 1 列波长，随后 10 列 = 10 个通道的 SPD）。
- 18 种材料的 f/P 模型，集中位于 `models/`：
  - `organic_models.py`（9 种有机颜料）、`inorganic_models.py`（3 种无机颜料，
    含对数定义域规则）—— 来自《公式集合_初版》；
  - `substrate_models.py`（6 种基材：新闻纸、竹纸、宣纸、麻纸、桑皮纸、绢）
    —— 来自《基材公式补充.docx》**正式公式**，其中"光谱响应率公式"实现为
    `response_*`（P(λ)）、"照度-时间公式"实现为 `quantity_*`（f(E,t)；
    E 即照度，与其余部分的 I 为同一物理量）。
  - **基材 pH 为固定属性**（新闻纸 5.0、竹纸 7.4、宣纸 8.5、麻纸 7.9、
    桑皮纸 8.3、绢 5.9）：已内置于公式中，**不是输入项**，界面仅作提示。
  - **f(E,t) 负值按 0 处理**（部分基材在低照度短时间下多项式为负）；
    P(λ) 负值**不裁剪**，按公式原值参与 K 积分。
  - 基材仍保留集中隔离结构（`SUBSTRATE_MODEL_SECTION_START/END`）：后续模型更新
    只需替换 `models/substrate_models.py` 内对应函数，无需修改界面、优化器或损伤链。
- 颜色指标（Rf/Rg/CCT/Duv）：主实现为 colour-science 的 TM-30-18（
  `models/color_metrics.py`），luxpy 为备用实现；两个实现已在测试中交叉校验。

---

## 4. 计算口径要点

- 候选光源：10 路权重线性叠加后最大值归一化（权重只承担光谱形状）。
- 光谱修正因子 K_m = ∫S_ω·P_m / ∫S_0·P_m（梯形积分，380–780 nm）。
- 参考工况 D0：D55、50 lx、100 h（与 pH 无关，pH 已内置于各基材公式）。
- 整体损伤 D_relic = Σ α_m·D_norm,m（α 为可见受光面积比例，Σα=1）。
- 强约束：Rf≥70、Rg≥88、2650 K≤CCT≤5550 K、|Duv|≤0.0054，不满足者为不可行解。
- 代表方案：保护优先 = 最小 D；展示优先 = 最大 F；综合折中 = 最小最大归一化
  偏差（minimax）：min_xk max[(D(xk)-Dmin)/(Dmax-Dmin), (Fmax-F(xk))/(Fmax-Fmin)]，
  即保护、展示两目标中表现较差者的归一化偏差尽量小（决议 R9，替代 v3 第 11
  节的欧氏距离规则）。
- **已知特性（决议 R17，按用户确认为公式固有行为，不做修正）**：六种基材的正式
  照度—时间公式 f(E,t) 在**照度 E 上非单调**（在时间 t 上单调递增）。例如 t=500 h
  时，新闻纸 f 由 50 lx 的 4.52 降至 250 lx 的 3.19，绢由 1 lx 的 10.14 降至
  300 lx 的 6.64。因此"保护优先"方案不保证对应最低照度；解读结果与论文表述时
  需注意这一点。

---

## 5. 部署（让其他人/其他设备使用）

### 5.1 局域网共享（最简单）
```powershell
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
同一网络内的设备访问 `http://<本机IP>:8501`（IP 用 `ipconfig` 查看，如 192.168.x.x）。
注意：Windows 防火墙需放行 8501 端口。

### 5.2 内网穿透（无需服务器，生成公网网址）
以 cpolar 为例：`cpolar http 8501`，得到形如 `https://xxx.cpolar.cn` 的公网地址，
任何地方可访问（免费额度适合临时演示）。

### 5.3 Docker / 云服务器（正式部署）
项目已内置 `Dockerfile`（含 Linux 中文字体）：
```bash
docker build -t heritage-lighting-optimizer .
docker run -d -p 8501:8501 --name heritage-opt heritage-lighting-optimizer
```
云主机（阿里云/腾讯云等）需在安全组放行 8501 端口，然后访问 `http://<公网IP>:8501`。
手动部署同样简单：装 Python 3.11 → `pip install -r requirements.txt` →
`streamlit run app.py --server.address 0.0.0.0 --server.port 8501`。

### 5.4 Streamlit Community Cloud（免费公网网址，需 GitHub）
项目已做好云部署准备（`packages.txt` 安装 Linux 中文字体、`.streamlit/config.toml`、
上传文件写入系统临时目录，兼容云平台只读文件系统）。操作：
1. 将项目**内容**（`app.py`、`requirements.txt`、`packages.txt`、`data/`、
   `models/`、`utils/`、`.streamlit/` 等，不含 `.venv`、`outputs/`）推送到 GitHub 仓库；
2. 登录 https://streamlit.io/cloud → New app → 选择该仓库、分支与入口文件 `app.py`；
3. 首次部署约 3–5 分钟（自动 `pip install -r requirements.txt` 并按 `packages.txt`
   安装中文字体），完成后得到固定网址 `https://<app名>.streamlit.app`。
4. 免费版说明：应用默认公开（可分享给任何人）；闲置后重启约需 1 分钟冷启动；
   若需"仅指定人员可访问"，在应用 Settings 中设为 Private（免费版有数量限制）。

> 说明：`Dockerfile` 与 `.streamlit/config.toml` 已就绪；Linux 服务器上 PNG 图表
> 中文由 `utils/export.py` 的跨平台字体注册保证（Docker 镜像已装 Noto CJK 字体）。

---

## 6. 目录结构

```
heritage_lighting_optimizer/
├─ app.py                  # Streamlit 主界面
├─ config.py               # 波长/参考工况/约束/NSGA-II 默认参数
├─ requirements.txt
├─ README.md
├─ data/                   # led_10_spd.xlsx, d55_spd.xlsx
├─ models/
│  ├─ material_registry.py # 17 种材料注册表与元数据
│  ├─ organic_models.py    # 9 种有机颜料 f/P
│  ├─ inorganic_models.py  # 3 种无机颜料 f/P（含对数定义域规则）
│  ├─ substrate_models.py  # 6 种基材正式公式 f(E,t)/P(λ) + 固定 pH（集中隔离区）
│  ├─ damage_chain.py      # K/D_raw/D0/D_norm/D_relic
│  ├─ display_model.py     # 冷/暖/中 F
│  ├─ color_metrics.py     # Rf/Rg/CCT/Duv（colour-science 主 / luxpy 备）
│  └─ optimizer.py         # NSGA-II、约束、三类代表方案
├─ utils/
│  ├─ spectral_io.py       # 光谱读取/校验/归一化
│  └─ export.py            # CSV/XLSX/PNG 导出
├─ tests/                  # pytest（验收 T1–T13）
└─ outputs/                # 结果输出目录
```
