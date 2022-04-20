Ссылка на bagfile - .

Чтобы записать bagfile, я воспользовался такой последовательностью команд:
```
dts start_gui_tools {autobot_name}
rosbag record /{autobot_name}/marked/roads/image/compressed  -O marked_roads.bag
```