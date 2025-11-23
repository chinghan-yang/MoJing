```shell
pip install ultralytics pygame
```

```shell
python yolo11_pose_video_ripple_cats_v5.py --bg_video ./calm_water.mp4 --cats ./cat0.png,./cat1.png,./cat2.png,./cat3.png,./cat4.png --cat_size_ratio 0.25 --cat_fade 3.0 --follow_smooth 0.18 --ripple_lambda 24 --ripple_speed 180 --ripple_amp 6 --ripple_radial_decay 0.015 --ripple_time_tau 1.6 --ripple_highlight 0.22 --camera 0 --weights yolo11n-pose.pt --draw_skeleton

```


```shell
python yolo11_pose_video_ripple_cats_v8.py --bg_video ./calm_water.mp4 --cats ./cat0.png,./cat1.png,./cat2.png,./cat3.png,./cat4.png --camera 0 --weights yolo11n-pose.pt --draw_skeleton

```