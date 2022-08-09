# simpleNEAT

最近闲得无聊想学学 `C++` 感觉好难啊，反正我自己是感觉写了个寂寞垃圾冷，之前一直想用 `Go` 实现一下这个算法，但是写错了导致占用了 `4G` 内存，干脆就重来。
`Python` 是不考虑的，国内把他吹成神了。
`NEAT` 算法的 `C++` 简单实现，还没有写种群。
线程池用的 `https://github.com/bshoshany/thread-pool/blob/master/BS_thread_pool.hpp`。

实验：

## xor实验（test_xor.cpp）

输出：
```bash
gen: 1 0x55b6062b9920 3 2 fitness: 0.75
gen: 37 0x55b6062b9b90 4 5 fitness: 0.75001
gen: 40 0x55b6062b9e90 4 5 fitness: 0.75001
gen: 41 0x55b6062b9710 4 5 fitness: 0.75013
gen: 42 0x55b6062b9c50 4 5 fitness: 0.77177
gen: 43 0x55b6062b9890 4 5 fitness: 0.99346
HiddenNeuronInnovations: 1
best: geration:43 fitness 0.99346 neurons 4 connections 5
neurons:
0 1
1 1
3 -1.18196
2 -2.8869
connections:
0 2 4.07129
0 3 -4.53845
1 2 -1.98597
1 3 4.98912
3 2 5.21552
predict:
0 0 [0] 7.7684e-07
1 1 [0] 0.0378004
0 1 [1] 0.842768
1 0 [1] 0.996992
```

可输出神经网络和可视化dot格式图像：
![image](https://user-images.githubusercontent.com/19900527/183606425-f4b5f56e-4f91-4b49-998f-7583573bdade.png)

## 鸢尾花实验（test_iris.cpp）

`Fitness` 达到 `0.979` 的神经网络有 `18` 个神经元和 `111` 条神经连接:
![image](https://user-images.githubusercontent.com/19900527/183610656-188c724e-5b41-4df4-8b3d-54ff73c88e5d.png)

## 基于 `sfml` 的分类实验（test_classification_sfml.cpp）

两个类型：
![image](https://user-images.githubusercontent.com/19900527/183608728-98a5ff85-7e7f-4553-a111-1e445bfc4899.png)

三个类型：
![image](https://user-images.githubusercontent.com/19900527/183608833-e566968f-fe3e-4f00-b4b7-906677e57b00.png)

四个类型：
![image](https://user-images.githubusercontent.com/19900527/183611655-2636c64e-f7f1-4934-8ff9-5e83072b129a.png)

以此类推...

## 基于 `sfml` 的寻路实验（test_pathfindder_sfml.cpp）

![image](https://user-images.githubusercontent.com/19900527/183612689-c0680195-5f40-4205-af9c-12b9a61f9346.png)

对于局部最优解的陷阱还没有找到高效的解决办法，个体会倾向于贴边走...
![image](https://user-images.githubusercontent.com/19900527/183613499-e3744e39-e6ba-4fe4-99d9-5563c333e0b9.png)

![image](https://user-images.githubusercontent.com/19900527/183615061-9525b28f-a3a0-491e-b6f2-38adf99da69b.png)

![image](https://user-images.githubusercontent.com/19900527/183615209-16fc9ae5-f06a-4660-8821-7bc43530f5ce.png)
