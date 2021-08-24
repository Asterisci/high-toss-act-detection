<!--
 * @Author: 范国藩
 * @Date: 2021-07-17 23:29:12
 * @Description: 
-->
# high-toss-act-detection

## Usage Demo

```
python main.py --video test2.mp4 --blank 20 100 25 340
```

### Output

#### command branch(no GUI)

- Output:

```python
# utils.py

out = {
    'start_id': $Frame_ID, 
    'posList': $posList
  }
```

- Video file: `$ID_output.mp4`

> 说明
> - `$Frame_ID`为视频开始的帧
> - `$posList`为从`$ID`开始往后每一帧中识别框`[x, y, w, h]`
> - `$ID`为识别到的物体id
>
> 回溯识别部分采用多线程调用，实际为异步返回结果
