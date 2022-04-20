Ссылка на bagfile - .

Чтобы записать bagfile, я воспользовался такой последовательностью команд:
```
dts start_gui_tools {autobot_name}
rosbag record /{autobot_name}/marked/roads/image/compressed -O marked_roads.bag
```

Чтобы проиграть bagfile, нужно ввести:
```
dts start_gui_tools {autobot_name}
rosbag play marked_roads.bag --topics /{autobot_name}/marked/roads/image/compressed
```

Чтобы посмотреть результат маркировки, нужно ввести:
```
dts start_gui_tools {autobot_name}
rqt_image_view
```

и в "выпадашке" выбрать топик /{autobot_name}/marked/roads/image/compressed.
