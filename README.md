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

# TODO

繪製rd圖 (記得改config和程式中的ckpt_name)

quality_level: {1,2,3,4}

lmbda: {0.0018, 0.0035, 0.0067, 0.013}

```
python examples/segmentation.py -c config/segmentation.yaml --quality_level 1 --lmbda 0.0018
python examples/segmentation.py -c config/segmentation.yaml -I --quality_level 1 --lmbda 0.0018
python examples/segmentation.py -c config/segmentation.yaml --quality_level 2 --lmbda 0.0035
python examples/segmentation.py -c config/segmentation.yaml -I --quality_level 2 --lmbda 0.0035
python examples/segmentation.py -c config/segmentation.yaml --quality_level 3 --lmbda 0.0067
python examples/segmentation.py -c config/segmentation.yaml -I --quality_level 3 --lmbda 0.0067
python examples/segmentation.py -c config/segmentation.yaml --quality_level 4 --lmbda 0.013
python examples/segmentation.py -c config/segmentation.yaml -I --quality_level 4 --lmbda 0.013

python draw_rd_chart.py
```

!!錯了
應該用不同config訓練再測試，baseline也要載不同等級的
iou應該算結果跟gt的，不是結果跟非壓縮的
config做到一半，tmux attach -t cal_rd有環境，剩下明天
```
python examples/segmentation.py -c segmentation_base1.yaml -I --ckptname "seg_base"
python examples/segmentation.py -c segmentation_base2.yaml -I --ckptname "seg_base"
python examples/segmentation.py -c segmentation_base3.yaml -I --ckptname "seg_base"
python examples/segmentation.py -c segmentation_base4.yaml -I --ckptname "seg_base"

python examples/segmentation.py -c segmentation_train1.yaml
python examples/segmentation.py -c segmentation_test1.yaml -I --ckptname "seg_mine1"
python examples/segmentation.py -c segmentation_train2.yaml
python examples/segmentation.py -c segmentation_test2.yaml -I --ckptname "seg_mine1"
python examples/segmentation.py -c segmentation_train3.yaml
python examples/segmentation.py -c segmentation_test3.yaml -I --ckptname "seg_mine1"
python examples/segmentation.py -c segmentation_train4.yaml
python examples/segmentation.py -c segmentation_test4.yaml -I --ckptname "seg_mine1"

python examples/segmentation.py -c segmentation_basecodic1.yaml -I --ckptname "seg_basecodic"
python examples/segmentation.py -c segmentation_basecodic2.yaml -I --ckptname "seg_basecodic"
python examples/segmentation.py -c segmentation_basecodic3.yaml -I --ckptname "seg_basecodic"
python examples/segmentation.py -c segmentation_basecodic4.yaml -I --ckptname "seg_basecodic"

python examples/segmentation.py -c segmentation_test1.yaml -I --ckptname "seg_mine_40"
python examples/segmentation.py -c segmentation_test2.yaml -I --ckptname "seg_mine_40"
python examples/segmentation.py -c segmentation_test3.yaml -I --ckptname "seg_mine_40"
python examples/segmentation.py -c segmentation_test4.yaml -I --ckptname "seg_mine_40"

python examples/segmentation.py -c segmentation_train1.yaml
python examples/segmentation.py -c segmentation_test1.yaml -I --ckptname "base_mine"
python examples/segmentation.py -c segmentation_train2.yaml
python examples/segmentation.py -c segmentation_test2.yaml -I --ckptname "base_mine"
python examples/segmentation.py -c segmentation_train3.yaml
python examples/segmentation.py -c segmentation_test3.yaml -I --ckptname "base_mine"
python examples/segmentation.py -c segmentation_train4.yaml
python examples/segmentation.py -c segmentation_test4.yaml -I --ckptname "base_mine"

python draw_rd_chart.py
```

找到原因，疑似是inference dataloader使用randomflip
不要再flip()了
剛把train的flip去掉，做了extract_feat的測試修改，正在train 1，剛剛跑出best有救了，加緊訓練其他的
val inf dataloader都改totensor, train改直接crop, 直接全部重畫
改有resize測試
加入summary writer, 每train 1 epoch紀錄loss一次
安裝tensorboard
真正的lambda是VPT_lmbda, lmbda只影響out_criterion['rd_loss']沒用到，但baseline的ckpt不同因此有鑑別
以防萬一，改了全測
pixel mean多減了(還用了BGR的) 重來
(RGB)->TransTIC壓縮->(RGB)->TransTIC Taskloss->(BGR)->RRSSM test pipeline torgb->(RGB)->RRSSM seg->(seg)
現在:
(RGB)->TransTIC壓縮->(RGB)->TransTIC Taskloss改->(RGB)->RRSSM test pipeline改->(RGB)->RRSSM seg->(seg)
改成直接crop
事實證明，儘管不完全變差，resize有效能下降的趨勢，Alan學長是對的
嘗試把train換成basecodic 1