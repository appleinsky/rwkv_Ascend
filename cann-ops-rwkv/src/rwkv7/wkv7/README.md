# WKV7 Vector

## 贡献说明
| 贡献者    | 贡献方  | 贡献算子                | 贡献时间      | 贡献内容                    |
|--------|------|---------------------|-----------|-------------------------|
| appleinsky | rwkv&昇腾 | wkv7 | 2025/7/13 | 新增wkv7算子 |

## 支持的产品型号

- Atlas A2训练系列产品
- Atlas 推理系列产品
- Atlas 200I/500 A2 推理产品
## 算子描述
- 功能描述
实现RWKV7的time_mixing单元的prefill wkv7算子计算逻辑:

### 原型信息
- `B`: 输入的Batch数 (shape: [], type: uint32)
- `T`: 输入的序列长度 (shape: [], type: uint32) 
- `C`: 输入的维度 (shape: [], type: uint32)
- `H`: 输入的block数, H能被C整除, N = C // H (shape: [], type: uint32)
  <table>
    <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">WKV7</td></tr>
    </tr>
    <tr><td rowspan="8" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td align="center">query</td><td align="center">B,H,T,N</td><td align="center">float</td><td align="center">ND</td></tr>
    <tr><td align="center">key</td><td align="center">B,H,T,N</td><td align="center">float</td><td align="center">ND</td></tr>
    <tr><td align="center">value</td><td align="center">B,H,T,N</td><td align="center">float</td><td align="center">ND</td></tr>
    <tr><td align="center">weight</td><td align="center">B,H,T,N</td><td align="center">float</td><td align="center">ND</td></tr>
    <tr><td align="center">a(-kk)</td><td align="center">B,H,T,N</td><td align="center">float</td><td align="center">ND</td></tr>
    <tr><td align="center">b(kk * a)</td><td align="center">B,H,T,N</td><td align="center">float</td><td align="center">ND</td></tr>
    <tr><td align="center">h0</td><td align="center">B,T,N,N</td><td align="center">float</td><td align="center">ND</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="2" align="center">算子输出</td><td align="center">output</td><td align="center">B,H,T,N</td><td align="center">float</td><td align="center">ND</td></tr>
    <td align="center">ht</td><td align="center">B,T,N,N</td><td align="center">float</td><td align="center">ND</td></tr>
    </tr>
  </table>

- 实现逻辑
```python
for t in range(T):
        for bi in range(B):
            for hi in range(H):
                q_t = q[bi, hi, t]
                k_t = k[bi, hi, t]
                v_t = v[bi, hi, t]
                a_t = a[bi, hi, t]
                b_t = b[bi, hi, t]
                w_t = np.exp(w[bi, hi, t])

                sa = np.sum((a_t[None, :] * h[bi, hi]), axis=1)

                h[bi, hi] = (h[bi, hi] * w_t[None, :] + 
                            k_t[None, :] * v_t[:, None] + 
                            sa[:, None] * b_t[None, :])

                y = np.sum((h[bi, hi] * q_t[None, :]), axis=1)
                o[bi, hi, t] = y
```

接口:

```cpp
extern "C" __global__ __aicore__ void wkv7(GM_ADDR k, GM_ADDR v, GM_ADDR w, GM_ADDR r, 
                                            GM_ADDR a, GM_ADDR b, GM_ADDR h0, GM_ADDR o, GM_ADDR ht, 
                                            GM_ADDR workspace, GM_ADDR tiling)
```

详细入参说明:
- `k`: 输入矩阵 k (shape: [B, H, T, N], datatype: float)
- `v`: 输入矩阵 v (shape: [B, H, T, N], datatype: float)
- `w`: 输入矩阵 w (shape: [B, H, T, N], datatype: float)
- `q`: 输入矩阵 r (shape: [B, H, T, N], datatype: float)
- `a`: 输入矩阵 a (shape: [B, H, T, N], datatype: float)
- `b`: 输入矩阵 b (shape: [B, H, T, N], datatype: float)
- `h0`: 输入矩阵 h (shape: [B, H, N, N], datatype: float)
- `o`: 输出矩阵 o (shape: [B, H, T, N], datatype: float)
- `ht`: 输出矩阵 o (shape: [B, H, N, N], datatype: float)

## 算子实现方案

### 分核, Tiling, 数据搬运

输入矩阵为 k, v, w, r, a, b, h0 其中 k, v, w, r, a, b 矩阵的 shape 为 [B, H, T, N], h0 矩阵的 shape 为 [B, H, N, N]; 输出矩阵为 o, ht。
因此选择在 B 和 H 维度进行分核, 对 k, v, w, r, o 在 T 维度进行单核 tiling 切分后循环tileNum次，每次搬运计算 tileLength*N 。

#### 分核

当前代码版本中支持 B * H 分核存在尾块的情况, 能整除的情况下每个核上处理 B * H/coreNum 份数据；非整除情况下大核多算一批的尾块。

#### Tiling

对数据排布为 [B, H, T, N] 顺序的 k, v, w, r, a, b, h0 矩阵, 在 T 维度切分后, 每次搬运的数据块大小为 tileLength * N 个元素。 具体 tileLength 可以根据 N 的大小、UB 内存（192KB、256KB）、DataCopy 指令单次搬运量越大连续 datablock 越长性能越好, 等综合考虑。

#### 其他 Tiling 方案

矩阵 k, v, w, r, a, b, h0 如果按照 [B, T, H, N] 的数据排布顺序, 则另一种方案是在对 H 维度进行 Tiling 切分, 每次搬运的数据块大小为 h_tileLength * N 个元素。但此种情况下, 由于计算过程需要同一个 state 矩阵 (N * N) 从 t=1 迭代到 t=T, 则在 UB 上需要存放 k * h_tileLength * N * N 份内存作为计算过程的中间变量 (k 为中间变量的个数)。如果 h_tileLength 过小, 则会导致搬运指令较过多影响性能。

因此综合考虑后选择将 H 维度与 T 维度对换, 并且选择在 T 维度进行切分, 此种情况下在 UB 上需要存放的中间变量为 k * N * N, tileLength 可以选择较大的值, 提高搬运指令性能和 mte2(gm->ub)带宽利用率。
