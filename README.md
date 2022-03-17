# RL_robotSim
Пробую обучать RL агента на задачу следования робота за лидером в симуляциях.

Лучшие конфиги:
- без препятствий и контроля скорости Configs/FollowContinuous/TD3_noObst_constSpeed.conf -> td3_algov1_explv1_noNorm
- без препятствий Configs/FollowContinuous/TD3_noObst.conf -> td3_algov1_explv1_noNorm, td3_archv4_algov1_explv1_noNorm, td3_archv5_algov1_explv1_noNorm
- с препятствиями Configs/FollowContinuous/TD3_obst.conf -> td3_explv5

