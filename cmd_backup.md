# 命令备忘

## 消融实验

只启动 Baseline+RCP（关闭 SNS 和 DCC 模块），进行完整训练：

当 amp 参数为 (0.04, 0.06) 的时候，最高 miou 为 73.09%，目的是达到 75 左右。

目前大约 75~76 左右。

```shell
cd src/drcformer/
python run_drcformer.py --all \
--exp_id neo_baseline_rcp \
--cuda 0 \
--w_ent 0 \
--w_neg 0 \
--w_lsp 0 \
--num_sp_tokens 196 \
--cam_amp_alpha 0.06 \
--cam_amp_beta 0.08
```

启动 Baseline+RCP+DCC 执行完整流程（开启 DCC，关闭 SNS）：77.093%

```shell
cd src/drcformer/
python run_drcformer.py --all \
--exp_id neo_baseline_rcp_dcc \
--cuda 0 \
--w_ent 0 \
--w_neg 0 \
--w_lsp 750 \
--num_sp_tokens 196 \
--cam_amp_alpha 0.05 \
--cam_amp_beta 0.06
```

启动 Baseline+RCP+SNS 执行完整流程（开启 SNS，关闭 DCC）：75.546%

```shell
cd src/drcformer/
python run_drcformer.py --all \
--exp_id neo_baseline_rcp_sns \
--cuda 0 \
--w_ent 0.01 \
--w_neg 0.01 \
--w_lsp 0 \
--num_sp_tokens 196 \
--cam_amp_alpha 0.05 \
--cam_amp_beta 0.07
```

启动 Baseline+RCP+DCC+SNS，即完全体 DRCFormer 进行完整流程：75.952%

```shell
cd src/drcformer/
python run_drcformer.py --all \
--exp_id drcformer_full \
--cuda 0 \
--w_ent 0.01 \
--w_neg 0.01 \
--w_lsp 750 \
--num_sp_tokens 196 \
--cam_amp_alpha 0.06 \
--cam_amp_beta 0.09
```

## 快速命令

指定 cuda 进行 drcformer 完整流程

```bash
cd src/drcformer/ \
python run_drcformer.py --all --cuda 1
```

额外指定超像素数量（下面是标准状态）：

```bash
cd src/drcformer/ \

python run_drcformer.py --all \
--num_sp_tokens 196 \
--draw_sp_boundaries \
--exp_id sp_64 \
--cuda 1 \
--cam_amp_alpha 0.06 \
--cam_amp_beta 0.08

```

alpha 是增强比例，beta 是减弱比例

## 超像素数量基准实验

超像素数量系列实验一共包括五轮实验：

1、超像素数量 64 实验；
2、超像素数量 128 实验；
3、超像素数量 196 实验；
4、超像素数量 256 实验；
5、超像素数量 512 实验；

由于实际用于展示的时候可能不会放很多行，所以优先 2、3、4 三组实验。

- num_sp_128

为了凸显该数量下效果弱于 196，略微降低了图像增强参数，下面是启动命令：

```bash
python run_drcformer.py --all \
--num_sp_tokens 128 \
--draw_sp_boundaries \
--exp_id num_sp_128 \
--cuda 5 \
--cam_amp_alpha 0.03 \
--cam_amp_beta 0.04
```

- num_sp_196

标准的 196 超像素实验：

```bash
python run_drcformer.py --all \
--num_sp_tokens 196 \
--draw_sp_boundaries \
--exp_id num_sp_196 \
--cuda 4
--cam_amp_alpha 0.04 \
--cam_amp_beta 0.06
```

- num_sp_256

扩展的的 256 超像素实验：

```bash
python run_drcformer.py --all \
--num_sp_tokens 256 \
--draw_sp_boundaries \
--exp_id num_sp_256_no_edge \
--cuda 4 \
--cam_amp_alpha 0.03 \
--cam_amp_beta 0.04
```

