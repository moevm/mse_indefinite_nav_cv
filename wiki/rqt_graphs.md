## Визуализация rqt_graph

Команда ```rosrun rqt_graph rqt_graph``` выводит запускает приложение, выводящее текущий rqt_graph. Приложение может обновляться без перезапуска.

## Чтение topic в реальном времени

Команда ```rostopic echo topic-name``` в реальном времени выводит все данные, проходящие через topic с именем topic-name

## Получение формата данных для конкретного топика

Команда ```rostopic type topic-name``` выводит тип данных, с которым работает topic с именем topic-name. 

Команда ```rostopic type topic-name | rosmsg show```  выводит информацию о всех полях типа данных, с которым работает topic-name.

### Пример

```
# dts start_gui_tools autobot-04
# rostopic list
/autobot04/auto_calibration_calculation_node/car_cmd
/autobot04/auto_calibration_node/car_cmd
/autobot04/camera_node/camera_info
/autobot04/camera_node/image/compressed
/autobot04/car_cmd_switch_node/cmd
/autobot04/client_count
/autobot04/connected_clients
/autobot04/coordinator_node/car_cmd
/autobot04/diagnostics/code/profiling
/autobot04/diagnostics/ros/links
/autobot04/diagnostics/ros/node
/autobot04/diagnostics/ros/parameters
/autobot04/diagnostics/ros/topics
/autobot04/duckiebot_il_lane_following/car_cmd
/autobot04/fsm_node/mode
/autobot04/joint_states
/autobot04/joy
...
# rostopic type /autobot04/joy
sensor_msgs/Joy
# rostopic type /autobot04/joy | rosmsg show
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
float32[] axes
int32[] buttons
```
