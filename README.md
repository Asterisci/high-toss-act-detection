# high-toss-act-detection

## Usage Demo

```
python main.py --video test2.mp4 --blank 20 100 25 340
```

### Input

- `--video`：输入文件
- `--min`：识别框最小size
- `--max`：识别框最大size
- `--blur`：高斯模糊范围
- `--target`：二值化阈值
- `--blank`：遮挡矩阵，用于遮挡时间水印条， 输入参数为`x y w h`

### Output

#### command branch(no GUI)

- Output:

```python
# utils.py

out = {
    'start_id': $Frame_ID, 
    'pos_list': $pos_list
  }
```

- Video file: `$ID_output.mp4`

> 说明
> - `$Frame_ID`为视频开始的帧
> - `$pos_list`为从`$ID`开始往后每一帧中识别框`[x, y, w, h]`
> - `$ID`为识别到的物体id
>
> 回溯识别部分采用多线程调用，实际为异步返回结果
