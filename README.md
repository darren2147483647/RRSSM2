# 專題內容

用TransTIC模型電腦視覺壓縮圖像來優化RRSSM河川切割模型的效能

TransTIC : https://github.com/NYCU-MAPL/TransTIC

RRSSM : https://github.com/nccudrone/RRSSM

# 程式

根目錄 : TransTIC/

- 訓練

進入conda環境

記得更改config/segmentation.yaml的ckpt與設置

```
python examples/segmentation.py -c config/segmentation.yaml
```

- 測試

進入conda環境

記得更改config/segmentation.yaml的ckpt與設置

```
python examples/segmentation.py -c config/segmentation.yaml -I
```

結果會在inference_img/
