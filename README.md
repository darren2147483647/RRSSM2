# RRSSM
**What is it?**
河川/道路影像切割模型(River/Road Semantic Segmentation Model)是切割圖像中河川/道路部分的模型，編寫語言是Python，計算資源需要不少於12G的GPU。
模型的權重沒有上傳上來，如果要測試，把`latest.pth`放到`[river/road/riverandroad]_model/latest.pth`的位置再執行。
**How to run**
* 先建立環境(建議conda)，`env.yml`包含所有所需套件
* 把測試圖片放到`test_image_folder/`裡
* 到根目錄執行以下指令。`river_model`可以換成`road_model`或`riverandroad_model`，指令中兩個地方要一起改。
    ```cmd
    python demo/predict.py --config river_model/config.py --checkpoint river_model/latest.pth
    ```
* 執行結果會出現在`demo/output/`

**My program stuck:(**
如果出現跟palette有關的錯誤，到`mmseg/core/evaluation/class_names.py`修改註解3類別/2類別的部分可能有作用?