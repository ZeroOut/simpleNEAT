#include <iostream>
#include <valarray>
#include <vector>
#include <map>
#include "raylib.h"
#include "lib/SimpleNEAT.hpp"

int screenWidth = 1000;  // 设置窗口宽度
int screenHeight = 1000;  // 设置窗口高度
int fps = 45;  // 训练时的fps限制
float wallMinimal = 15.f;
float initRotation[4] = {0.f, 0.f, 0.f, 0.f};  // 个体的初始化旋转角度
float objectSize = 10.f;;  // 个体的半径
float sensorMax = objectSize * 6.f;  // 距离传感器长度
float stepSize = 15.f;  // 移动和旋转的步长
bool isStart = false;  // 是否开始训练
int frameLimit = 1500;  // 每一代帧数限制
const int sensorCount = 11;  // 具体传感器数量
int skipDrawCount = 1; // 不需要渲染的数量
int forceKeepPerZone = 8; // 强制每个区域保留到达目标的数量
bool canExit = false;
Vector2 initPosistion[4] = {{25.f,                      25.f},
                            {float(screenWidth) - 25.f, 25.f},
                            {float(screenWidth) - 25.f, float(screenHeight) - 25.f},
                            {25.f,                      float(screenHeight) - 25.f}};  // 个体出生位置，初始在窗口从左到右从上往下 100,100
Vector2 targetPosition[4] = {{float(screenWidth) / 2 - 25.f, float(screenHeight) / 2 - 25.f},
                             {float(screenWidth) / 2 + 25.f, float(screenHeight) / 2 - 25.f},
                             {float(screenWidth) / 2 + 25.f, float(screenHeight) / 2 + 25.f},
                             {float(screenWidth) / 2 - 25.f, float(screenHeight) / 2 + 25.f}};  // 目标位置，初始在窗口从右到左从下往上 100,100
znn::SimpleNeat sneat;  // 创建NEAT对象

void initNeat() {  // 初始化神经网络和种群
    znn::Opts.InputSize = sensorCount + 2;  // 设置神经网络输入节点数量=传感器数量+个体与目标的相对距离+个体朝向与目标的相对角度
    znn::Opts.OutputSize = 2;  // 设置神经网络输出节点数量
    znn::Opts.ActiveFunction = znn::Sigmoid;  // 使用的激活函数
    znn::Opts.IterationTimes = 0;  // 迭代次数，0为不限制
    znn::Opts.FitnessThreshold = 0.f;  // 个体适应值阈值，0为不限制
    znn::Opts.IterationCheckPoint = 10;  // 保存最优神经网络的迭代次数，1为每代保存
    znn::Opts.ThreadCount = 16;  // 多线程数量，不设置则为设备默认数量
    znn::Opts.MutateAddNeuronRate = 0.15f;  // 添加新神经元的概率
    znn::Opts.MutateAddConnectionRate = 1.f;  // 添加新连接的概率
    znn::Opts.PopulationSize = 200;  // 训练个体的数量
    znn::Opts.NewSize = 0;  // 每一代新生个体的数量
    znn::Opts.ChampionToNewSize = 180;  // 冠军被复制和交配的目标数量
    znn::Opts.ChampionKeepSize = 60;  // 冠军的数量
    znn::Opts.KeepWorstSize = 0;  // 保留最差个体的数量
    znn::Opts.KeepComplexSize = 1;  // 保留最复杂神经网络个体的数量，用于交配产生更复杂的神经网络
    znn::Opts.WeightRange = 12;  // 神经连接的权重范围，-12至12
    znn::Opts.BiasRange = 6;  // 神经元的偏置范围，-6至6
    znn::Opts.MutateBiasRate = .1f;  // 神经元的偏置变异概率
    znn::Opts.MutateWeightRate = .1f;  // 神经元连接的权重变异概率
    znn::Opts.MutateBiasDirectOrNear = .9f;  // 神经元偏置随机变异和就近变异的比例
    znn::Opts.MutateWeightDirectOrNear = .9f;  // 神经元连接权重随机变异和就近变异的比例
    znn::Opts.Enable3dNN = false;  // 是否显示3d实时可视化神经网络，不能启用，因为用的raylib库，训练环境也用的raylib库，不能开启多窗口
    znn::Opts.CheckPointPath = "/tmp/raylib_path_findder";  // 自动保存神经网络和NEAT创新ID的路径
    znn::Opts.StartWithFCNN = true;
    znn::Opts.FCNN_hideLayers = {16,16};

    srandom((unsigned) clock());  // 初始化随机种子

    sneat.Start();  // 初始化NEAT神经网络和种群，如果自动保存路径存在，则导入，不存在则新建
}

Vector2 getXY(float angle, float distance) {  // 通过角度和距离计算坐标
    Vector2 result;
    float radians = angle * PI / 180.f;
    result.x = distance * std::cos(radians);
    result.y = distance * std::sin(radians);
    return result;
}

struct myWall {  // 障碍物
    std::vector<Vector2> path;  // 存储坐标的容器

    void add() {  // 添加坐标
        Vector2 mousePos = GetMousePosition();
        if (path.empty() || (!path.empty() && std::abs(path[path.size() - 1].x - float(mousePos.x)) > wallMinimal || std::abs(path[path.size() - 1].y - float(mousePos.y)) > wallMinimal)) {  // 判断是否和上一个坐标x或者y相差至少
            path.push_back(mousePos);
        }
    }

    void draw() {  // 绘制障碍
        if (!path.empty()) {
            for (int i = 1; i < path.size(); ++i) {
                DrawLineEx(path[i - 1], path[i], 1.f, WHITE);
            }
        }
    }
};

std::vector<std::vector<myWall>> walls;  // 存储多个障碍

myWall createScreenWall(int zone) {  // 创建窗口四周的障碍
    myWall aWall;

    if (zone == 0) {
        aWall.path.push_back({0, 0});
        aWall.path.push_back({0, float(screenHeight) / 2});
        aWall.path.push_back({float(screenWidth) / 2, float(screenHeight) / 2});
        aWall.path.push_back({float(screenWidth) / 2, 0});
        aWall.path.push_back({0, 0});
    } else if (zone == 1) {
        aWall.path.push_back({float(screenWidth) / 2, 0});
        aWall.path.push_back({float(screenWidth) / 2, float(screenHeight) / 2});
        aWall.path.push_back({float(screenWidth), float(screenHeight) / 2});
        aWall.path.push_back({float(screenWidth), 0});
        aWall.path.push_back({float(screenWidth) / 2, 0});
    } else if (zone == 2) {
        aWall.path.push_back({float(screenWidth) / 2, float(screenHeight) / 2});
        aWall.path.push_back({float(screenWidth) / 2, float(screenHeight)});
        aWall.path.push_back({float(screenWidth), float(screenHeight)});
        aWall.path.push_back({float(screenWidth), float(screenHeight) / 2});
        aWall.path.push_back({float(screenWidth) / 2, float(screenHeight) / 2});
    } else if (zone == 3) {
        aWall.path.push_back({0, float(screenHeight) / 2});
        aWall.path.push_back({0, float(screenHeight)});
        aWall.path.push_back({float(screenWidth) / 2, float(screenHeight)});
        aWall.path.push_back({float(screenWidth) / 2, float(screenHeight) / 2});
        aWall.path.push_back({0, float(screenHeight) / 2});
    }

    return aWall;
}

bool getCollion(Vector2 center, Vector2 sensorTail, Vector2 &collisionPoint, float &sensorDistance, int belongZone) {  // 根据两条线的起止坐标判断是否相交
    std::map<float, Vector2> dis2Pos;
    bool isCollision = false;

    for (auto &w: walls[belongZone]) {
        for (int i = 1; i < w.path.size(); ++i) {
            Vector2 collisionPos;
            if (CheckCollisionLines(center, sensorTail, w.path[i - 1], w.path[i], &collisionPos)) {
                float distance = std::sqrt(std::pow(collisionPos.x - center.x, 2.f) + std::pow(collisionPos.y - center.y, 2.f));
                dis2Pos[distance] = collisionPos;
                isCollision = true;
            }
        }
    }

    collisionPoint = dis2Pos.begin()->second;
    sensorDistance = dis2Pos.begin()->first;
    return isCollision;
}

struct object {  // 训练个体
    float rotation;  // 旋转角度
    Vector2 position;  // 出生位置
    bool isDead = false;  // 是否死亡
    float speed = 0.f;  // 速度
    std::vector<Vector2> path;  // 走过的路径
    std::vector<float> pathWidth;  // 走过路径对应的路宽，根据速度判断
    Vector2 sensorsPos[sensorCount]{};  // 距离传感器的相对末端坐标
    Vector2 sensorCol[sensorCount]{};  // 距离传感器的相对探测到障碍的交叉坐标
    float sensorDis[sensorCount];  // 距离传感器到障碍的长度
    float score = 0.f;  // 记录得分
    float targetAngle;  // 个体朝向和目标的相对角度
    float targetDistance;  // 个体和目标的距离
    float beginDistance = 0.f;  // 个体和出生位置的距离
    Vector2 targetPos;
    int belongZone;
    int isWin = false;

    void setSensors() {  // 放置具体传感器
        for (int i = 0; i < sensorCount; ++i) {
            Vector2 sensorTail = getXY(float(i) * 30.f + rotation - 150.f, sensorMax);
            sensorsPos[i].x = sensorTail.x + position.x;
            sensorsPos[i].y = sensorTail.y + position.y;
        }
    }

    void getSensorsInfo() {  // 更新传感器数据
        for (int i = 0; i < sensorCount; ++i) {
            if (!getCollion(position, sensorsPos[i], sensorCol[i], sensorDis[i], belongZone)) {
                sensorCol[i] = sensorsPos[i];  // 传感器与障碍物交叉的位置，没有交叉则为传感器目标位置
                sensorDis[i] = sensorMax;  // 传感器与障碍之间的距离，没有交叉则为预设最大值
            }

            if (sensorDis[i] < objectSize) {
                isDead = true;  // 如果传感器道障碍的距离小于个体半径，则判断为死亡
                break;
            }
        }

        if (std::abs(position.x - targetPos.x) < 10.f && std::abs(position.y - targetPos.y) < 10.f) {  // 判断个体是否到达目标坐标
            isDead = true;  // 到达坐标则死亡
            isWin = true;
        }

        if (path.size() > 2) {  // 判断个体是否走了老路，通过路径记录和碰撞判断
            for (int i = 1; i < path.size() - 2; ++i) {
                if (path[i].x != path[i - 1].y || path[i].x != path[i - 1].y) {
                    if (CheckCollisionLines(path[path.size() - 1], path[path.size() - 2], path[i], path[i - 1], nullptr)) {
                        isDead = true;  // 如果和自己的运动轨迹碰撞则死亡
                        break;
                    }
                }
            }
        }

        targetDistance = std::sqrt(std::pow(position.x - targetPos.x, 2.f) + std::pow(position.y - targetPos.y, 2.f));  // 更新个体和目标的距离
        if (targetDistance < 1.f) {  // 为便于分数判定，需要将目标距离作为被除数
            targetDistance = 1.f;  // 如果距离小于1则为1，避免分数特别大
        }
        targetAngle = std::atan2((targetPos.y - position.y), (targetPos.x - position.x)) / PI * 180.f - 180.f;  // 更新个体朝向与目标的相对角度
        while (targetAngle < -180.f) {
            targetAngle += 360.f;
        }
        while (targetAngle > 180.f) {
            targetAngle -= 360.f;
        }

//        if (std::abs(targetAngle) < 10.f) {  // 如果相对角度小于10，则加分
//            score += 1.f;
//        }
    }

    object(Vector2 p, float r, Vector2 tp, int bz) {  // 创建个体时的初始化操作
        belongZone = bz;
        targetPos = tp;
        position = p;
        rotation = r;
        setSensors();  // 更新传感器位置
        getSensorsInfo();  // 更新传感器数据
    }

    void rotate(float angle) {  // 个体旋转操作
        rotation = float(int((rotation + angle * stepSize / 3.f) * 10) % 3600) / 10.f;
//        if ((sensorDis[2] < objectSize * 1.1f && sensorDis[2] < sensorDis[8] && angle > 0) || (sensorDis[8] < objectSize * 1.1f && sensorDis[2] > sensorDis[8] && angle < 0)) {  // 判断是否通过旋转规避近距离障碍
//            score += 1.f;
//        }
        setSensors();
        getSensorsInfo();
    }

    void move(float distance) {  // 个体移动操作
        speed = distance * objectSize * 1.5f;  // 更新速度，用于可视化尾喷长度
        Vector2 movePos = getXY(rotation, distance * stepSize);  // 获取需要移动的相对坐标
        position.x += movePos.x;  // 更新个体坐标x
        position.y += movePos.y;  // 更新个体坐标y
        if (distance > 0.f) {  // 如果个体移动距离太小，则判断为死亡
            path.push_back(position);
            if (distance > 0.5f) {  // 为防止可视化路径的时候宽度太小，则设置最小宽度0.3
                pathWidth.push_back(distance);
//                score += distance;  // 更新分数，叠加速度
            } else {
                pathWidth.push_back(0.5f);
            }
        } else {
//            isDead = true;
        }

        setSensors();  // 更新传感器位置
        getSensorsInfo();  // 更新传感器数据
    }

    void draw() {  // 绘制个体
        if (path.size() > 1) {  // 绘制个体移动路径，线条需要两个坐标
            for (int i = 1; i < path.size(); ++i) {
                DrawLineEx(path[i - 1], path[i], 1., ColorAlpha(GREEN, (1.5f - pathWidth[i]) * 0.3f));  // 路径宽度改为路径透明度由宽度判定
            }
        }

//            if (!isDead) {  // 如果存活则绘制尾喷
        Vector2 tailPos0 = getXY(rotation + 180.f, speed + objectSize / 2.f);  // 获取尾喷相对位置用于绘制，长度由速度决定
        Vector2 tailPos1 = getXY(rotation + 135.f, objectSize / 2.f);  // 获取尾喷相对位置用于绘制
        Vector2 tailPos2 = getXY(rotation - 135.f, objectSize / 2.f);  // 获取尾喷相对位置用于绘制
        tailPos0.x += position.x;
        tailPos0.y += position.y;
        tailPos1.x += position.x;
        tailPos1.y += position.y;
        tailPos2.x += position.x;
        tailPos2.y += position.y;
        DrawTriangle(tailPos0, tailPos1, tailPos2, ColorAlpha(YELLOW, 0.6f));
//                DrawLineEx({tailPos0.x + position.x, tailPos0.y + position.y}, {tailPos1.x + position.x, tailPos1.y + position.y}, objectSize * .3f,ColorAlpha(YELLOW, pathWidth[pathWidth.size() - 1]));
//            }

        auto objColor = WHITE;  // 如果个体存活，则本体为白色，死亡为红色

        if (!isDead) {  // 如果个体存活则绘制传感器
            for (auto sc: sensorCol) {
                DrawLineEx(position, sc, 1, ColorAlpha(BLUE, 0.5f));
                DrawCircleV(sc, objectSize / 10.f * 3.f, ColorAlpha(RED, 0.5f));
            }
        } else {
            objColor = RED;
        }

        DrawPolyLinesEx(position, 3, objectSize, rotation + 30.f, objectSize / 5.f, objColor);  // 绘制个体本体，三角形
        Vector2 headPos = getXY(rotation, objectSize);  // 获取头部坐标用于给个体头部画一根线
        DrawLineEx(position, {headPos.x + position.x, headPos.y + position.y}, 1, objColor);  // 给个体头部画一根线分辨方向
    }
};

int lastWallZone = -1;

void keyControl() {  // 用户输入控制
    if (IsMouseButtonDown(0)) {  // 鼠标左键绘制障碍
        auto mousePosistion = GetMousePosition();
        int thisWallZone;
        if (mousePosistion.x <= float(screenWidth) / 2 && mousePosistion.y <= float(screenHeight) / 2) {
            thisWallZone = 0;
        } else if (mousePosistion.x >= float(screenWidth) / 2 && mousePosistion.y <= float(screenHeight) / 2) {
            thisWallZone = 1;
        } else if (mousePosistion.x >= float(screenWidth) / 2 && mousePosistion.y >= float(screenHeight) / 2) {
            thisWallZone = 2;
        } else if (mousePosistion.x <= float(screenWidth) / 2 && mousePosistion.y >= float(screenHeight) / 2) {
            thisWallZone = 3;
        }

        if (lastWallZone != thisWallZone && lastWallZone != -1) {
            walls[lastWallZone].push_back(myWall{});
        } else {
            walls[thisWallZone][walls[thisWallZone].size() - 1].add();
        }

        lastWallZone = thisWallZone;
    }


    if (IsMouseButtonReleased(0)) {  // 鼠标左键抬起用于添加新的障碍物列表，避免只绘制一条线
        auto mousePosistion = GetMousePosition();
        lastWallZone = -1;
        if (mousePosistion.x <= float(screenWidth) / 2 && mousePosistion.y <= float(screenHeight) / 2) {
            walls[0].push_back(myWall{});
        } else if (mousePosistion.x >= float(screenWidth) / 2 && mousePosistion.y <= float(screenHeight) / 2) {
            walls[1].push_back(myWall{});
        } else if (mousePosistion.x >= float(screenWidth) / 2 && mousePosistion.y >= float(screenHeight) / 2) {
            walls[2].push_back(myWall{});
        } else if (mousePosistion.x <= float(screenWidth) / 2 && mousePosistion.y >= float(screenHeight) / 2) {
            walls[3].push_back(myWall{});
        }
    }

    if (IsKeyPressed('C')) {  // C键清除障碍
        auto mousePosistion = GetMousePosition();
        int thisWallZone;
        if (mousePosistion.x <= float(screenWidth) / 2 && mousePosistion.y <= float(screenHeight) / 2) {
            thisWallZone = 0;
        } else if (mousePosistion.x >= float(screenWidth) / 2 && mousePosistion.y <= float(screenHeight) / 2) {
            thisWallZone = 1;
        } else if (mousePosistion.x >= float(screenWidth) / 2 && mousePosistion.y >= float(screenHeight) / 2) {
            thisWallZone = 2;
        } else if (mousePosistion.x <= float(screenWidth) / 2 && mousePosistion.y >= float(screenHeight) / 2) {
            thisWallZone = 3;
        }

        walls[thisWallZone].clear();
        walls[thisWallZone].push_back(createScreenWall(thisWallZone));
        walls[thisWallZone].push_back(myWall{});
    }

    if (IsMouseButtonPressed(1)) {  // 鼠标右键清除上一障碍
        auto mousePosistion = GetMousePosition();
        int thisWallZone;
        if (mousePosistion.x <= float(screenWidth) / 2 && mousePosistion.y <= float(screenHeight) / 2) {
            thisWallZone = 0;
        } else if (mousePosistion.x >= float(screenWidth) / 2 && mousePosistion.y <= float(screenHeight) / 2) {
            thisWallZone = 1;
        } else if (mousePosistion.x >= float(screenWidth) / 2 && mousePosistion.y >= float(screenHeight) / 2) {
            thisWallZone = 2;
        } else if (mousePosistion.x <= float(screenWidth) / 2 && mousePosistion.y >= float(screenHeight) / 2) {
            thisWallZone = 3;
        }

        if (walls[thisWallZone].size() > 2) {
            walls[thisWallZone].pop_back();
            walls[thisWallZone][walls[thisWallZone].size() - 1].path.clear();
        }
    }

    if (IsKeyDown('B')) {  // B键用于设置个体出生位置
        auto mousePosistion = GetMousePosition();
        if (mousePosistion.x <= float(screenWidth) / 2 && mousePosistion.y <= float(screenHeight) / 2) {
            initPosistion[0] = mousePosistion;
            initRotation[0] = std::atan2(targetPosition[0].y - initPosistion[0].y, targetPosition[0].x - initPosistion[0].x) / PI * 180.f;  // 设置完出生位置后更新个体初始朝向
        } else if (mousePosistion.x >= float(screenWidth) / 2 && mousePosistion.y <= float(screenHeight) / 2) {
            initPosistion[1] = mousePosistion;
            initRotation[1] = std::atan2(targetPosition[1].y - initPosistion[1].y, targetPosition[1].x - initPosistion[1].x) / PI * 180.f;  // 设置完出生位置后更新个体初始朝向
        } else if (mousePosistion.x >= float(screenWidth) / 2 && mousePosistion.y >= float(screenHeight) / 2) {
            initPosistion[2] = mousePosistion;
            initRotation[2] = std::atan2(targetPosition[2].y - initPosistion[2].y, targetPosition[2].x - initPosistion[2].x) / PI * 180.f;  // 设置完出生位置后更新个体初始朝向
        } else if (mousePosistion.x <= float(screenWidth) / 2 && mousePosistion.y >= float(screenHeight) / 2) {
            initPosistion[3] = mousePosistion;
            initRotation[3] = std::atan2(targetPosition[3].y - initPosistion[3].y, targetPosition[3].x - initPosistion[3].x) / PI * 180.f;  // 设置完出生位置后更新个体初始朝向
        }
    }

    if (IsKeyDown('T')) {  // T键用于设置目标位置
        auto mousePosistion = GetMousePosition();
        if (mousePosistion.x <= float(screenWidth) / 2 && mousePosistion.y <= float(screenHeight) / 2) {
            targetPosition[0] = mousePosistion;
            initRotation[0] = std::atan2(targetPosition[0].y - initPosistion[0].y, targetPosition[0].x - initPosistion[0].x) / PI * 180.f;  // 设置完出生位置后更新个体初始朝向
        } else if (mousePosistion.x >= float(screenWidth) / 2 && mousePosistion.y <= float(screenHeight) / 2) {
            targetPosition[1] = mousePosistion;
            initRotation[1] = std::atan2(targetPosition[1].y - initPosistion[1].y, targetPosition[1].x - initPosistion[1].x) / PI * 180.f;  // 设置完出生位置后更新个体初始朝向
        } else if (mousePosistion.x >= float(screenWidth) / 2 && mousePosistion.y >= float(screenHeight) / 2) {
            targetPosition[2] = mousePosistion;
            initRotation[2] = std::atan2(targetPosition[2].y - initPosistion[2].y, targetPosition[2].x - initPosistion[2].x) / PI * 180.f;  // 设置完出生位置后更新个体初始朝向
        } else if (mousePosistion.x <= float(screenWidth) / 2 && mousePosistion.y >= float(screenHeight) / 2) {
            targetPosition[3] = mousePosistion;
            initRotation[3] = std::atan2(targetPosition[3].y - initPosistion[3].y, targetPosition[3].x - initPosistion[3].x) / PI * 180.f;  // 设置完出生位置后更新个体初始朝向
        }
    }

    if (IsKeyPressed(KEY_SPACE)) {  // 空格键用于控制是否开始训练
        if (isStart) {
            isStart = false;
            SetTargetFPS(60);  // 没开始训练时帧率限制为30
        } else {
            isStart = true;
            SetTargetFPS(fps);
        }
    }

    if (IsKeyPressed('N')) {
        sneat.population.CreatePopulationByGiving();
    }
}

bool isBreakFunc() {  // 用于NEAT训练循环中判断是否中断
    return !isStart;
};

bool cmp(std::pair<int, float> &a, std::pair<int, float> &b) {
    return a.second > b.second;// 从大到小排列
}

std::vector<int> OrderByScore(std::map<int, float> &M) {  // Comparator function to sort pairs according to second value
    std::vector<int> result;
    std::vector<std::pair<int, float> > A;// Declare vector of pairs
    for (auto &it: M) {  // Copy key-value pair from Map to vector of pairs
        A.push_back(it);
    }
    std::sort(A.begin(), A.end(), cmp);// Sort using comparator function
    for (auto &it: A) {
        result.push_back(it.first);
    }
    return result;
}

int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT);  // 设置抗锯齿

    InitWindow(screenWidth, screenHeight, "寻路实验");  // 初始化raylib窗口

    SetTargetFPS(60);  // 设置帧率

    for (int i = 0; i < 4; ++i) {
        walls.push_back({createScreenWall(i)});
        walls[i].push_back(myWall{});
        initRotation[i] = std::atan2(targetPosition[i].y - initPosistion[i].y, targetPosition[i].x - initPosistion[i].x) / PI * 180.f;  // 初始化个体朝向
    }

    initNeat();  // 初始化神经网络和种群

    int stepCount = 0;  // 训练迭代计数器
    std::function<std::unordered_map<znn::NetworkGenome *, float>()> interactiveFunc = [&]() {
        ++stepCount;

        std::vector<std::vector<object>> objs;  // 新建个体集容器
        for (int i = 0; i < znn::Opts.PopulationSize; ++i) {  // 塞满个体
            std::vector<object> thisObjs;
            for (int ii = 0; ii < 4; ++ii) {
                object obj(initPosistion[ii], initRotation[ii], targetPosition[ii], ii);
                thisObjs.push_back(obj);
            }
            objs.push_back(thisObjs);
        }

        for (int step = 0; step < frameLimit; ++step) {  // 每一代训练，基于帧数限制的循环
            keyControl();  // 用户输入控制

            int deadCount = 0;  // 死亡个体计数器
            std::vector<std::future<void>> thisFuture;

            for (int iii = 0; iii < znn::Opts.PopulationSize; ++iii) {  // 每个训练个体的神经网络判断输入和输出
                thisFuture.push_back(znn::tPool.submit([&](int i) {
                    for (int ii = 0; ii < 4; ++ii) {
                        if (!objs[i][ii].isDead) {  // 如果个体存活则继续
                            if ((step > 50 && objs[i][ii].path.size() < 20) || (objs[i][ii].path.size() > 10 && std::abs(objs[i][ii].path[objs[i][ii].path.size() - 1].y - objs[i][ii].path[objs[i][ii].path.size() - 10].y) < 1.f &&
                                                                                std::abs(objs[i][ii].path[objs[i][ii].path.size() - 1].x - objs[i][ii].path[objs[i][ii].path.size() - 10].x) < 1.f) || objs[i][ii].position.x < 0 || objs[i][ii].position.x > float(screenWidth) ||
                                objs[i][ii].position.y < 0 || objs[i][ii].position.y > float(screenHeight)) {  // 简单判断死亡
                                objs[i][ii].isDead = true;
                            } else {
                                std::vector<float> perInputs;  // 准备神经网络输入数据

                                for (auto &sd: objs[i][ii].sensorDis) {  // 输入数据放入传感器到障碍物的距离
                                    perInputs.push_back(1.f - ((sd - objectSize) / (sensorMax - objectSize)));  // 距离除以传感器最大值，离得越近数值越大，同时排除个体自身尺寸，使得输入值在0-1范围
                                }
                                perInputs.push_back(objs[i][ii].targetAngle / 180.f);  // 输入数据放入个体朝向到目标的相对角度
                                if (objs[i][ii].targetDistance < sensorMax) { // 输入数据放入个体和目标的距离
//                                    perInputs.push_back(1.f - objs[i][ii].targetDistance / sensorMax);
                                    perInputs.push_back(1.f);
                                } else {
                                    perInputs.push_back(0.f);
                                }

                                std::stringstream debug;
                                debug << objs[i][ii].targetDistance;
                                if (debug.str() == "nan" || debug.str() == "-nan") {
                                    std::cout << "DEBUG " << &objs[i][ii].targetDistance << ", " << nullptr;
                                    std::cout << "DEBUG";
                                    exit(100);
                                }

                                std::vector<float> nextMove = sneat.population.generation.neuralNetwork.FeedForwardPredict(&sneat.population.NeuralNetworks[i], perInputs, false);  // 根据输入数据计算每个神经网络的输出
                                objs[i][ii].rotate((nextMove[0] - 0.5f) * 2.f);  // 执行输出结果的旋转操作
                                objs[i][ii].move(nextMove[1]);  // 执行输出结果的移动操作

//                                if (perInputs[(sensorCount - 1) / 2] > .5f && nextMove[1] < .9f) {  // 简单判断个体前方有障碍则减速的加分
//                                    objs[i][ii].score += 1.f;
//                                }
                            }
                        } else {
                            if (objs[i][ii].isWin) {
                                objs[i][ii].score = 1.f / float(objs[i][ii].path.size());
                            } else {
                                objs[i][ii].score = -objs[i][ii].targetDistance / std::sqrt(std::pow(objs[i][ii].targetPos.x - initPosistion[ii].x, 2.f) + std::pow(objs[i][ii].targetPos.y - initPosistion[ii].y, 2.f));
                            }
                            //                        for (int iii = 0; iii < 4; ++iii) {
                            //                            objs[i][iii].isDead = true;  // 如果一个死亡，统一神经网络的四个一起死亡
                            //                        }
                            znn::mtx.lock();
                            ++deadCount;
                            znn::mtx.unlock();
                        }
                    }
                }, iii));
            }

            for (auto &f: thisFuture) {
                f.wait();
            }

            if (WindowShouldClose()) {
                canExit = true;
                isStart = false;
            }

            if (stepCount % skipDrawCount == 0) {  // 如果达到自动保存次数，则可视化显示
                BeginDrawing();  // 开始绘制
                ClearBackground(BLACK);  // 清空背景

                for (int i = 0; i < 4; ++i) {
                    DrawCircleV(initPosistion[i], 10.f, GRAY);  // 绘制出生点
                    DrawCircleV(targetPosition[i], 10.f, RED);  // 绘制目标点
                    for (auto &w: walls[i]) {  // 绘制障碍
                        w.draw();
                    }
                }

                for (int i = 0; i < znn::Opts.PopulationSize; ++i) {
                    for (int ii = 0; ii < 4; ++ii) {
                        objs[i][ii].draw();  // 绘制个体
                    }
                }

                if (canExit || !isStart) {
                    DrawText("Waiting for training done.", screenWidth / 4, screenHeight / 2 - screenWidth / 25 / 2, screenWidth / 25, GRAY);
                }

                DrawFPS(screenWidth / 2 + 10, 10);  // 绘制fps
                EndDrawing();  // 单帧绘制完毕
            }

            if (deadCount == znn::Opts.PopulationSize * 4) {  // 如果全部个体死亡则终止本代
                break;
            }
        }

        int winnerCount[5] = {0, 0, 0, 0, 0};  // 到达目标的个体计数器，记录同一个神经网络达到0，1，2，3，4个目标的数量
        int winnerPerZoneCount[4] = {0, 0, 0, 0};  // 每个区域到达目标的个体计数器，记录同一个神经网络达到0，1，2，3区域的数量
        std::vector<std::map<int, float>> perZoneScores(4);

        for (int i = 0; i < znn::Opts.PopulationSize; ++i) { // 统计
            int perNNwinnerCount = 0;

            for (int ii = 0; ii < 4; ++ii) {
                if (objs[i][ii].isWin) {
                    ++perNNwinnerCount;
                    ++winnerPerZoneCount[ii];
                }
                perZoneScores[ii][i] = objs[i][ii].score;
            }

            ++winnerCount[perNNwinnerCount];
        }

        std::cout << "Score:";
        for (int ii = 0; ii < 4; ++ii) {
            auto orderedScoreIndex = OrderByScore(perZoneScores[ii]);

            for (int f = 0; f < forceKeepPerZone; ++f) { // 强制每个区域到达目标的个体保留x个
                objs[orderedScoreIndex[f]][ii].score += 10.f;
            }

            std::cout << " [" << ii << "] " << objs[orderedScoreIndex[0]][ii].score << ",";
        }

        std::unordered_map<znn::NetworkGenome *, float> populationFitness;  // 创建神经网络地址对应的适应度map

        for (int i = 0; i < znn::Opts.PopulationSize; ++i) {  // 更新每个个体的适应度（得分）
            populationFitness[&sneat.population.NeuralNetworks[i]] = 0.f;
            for (int ii = 0; ii < 4; ++ii) {
                populationFitness[&sneat.population.NeuralNetworks[i]] += objs[i][ii].score;
            }
        }

//        if (winnerCount[4] > 0) {  // 如果有同时达到四个区域目标的神经网络，则结束训练
//            isStart = false;
//        }

        std::cout << "\nWinner:";
        for (int i = 4; i >= 0; --i) {  // 重置达到目标的个体计数
            std::cout << " [" << i << "] " << winnerCount[i] << ",";
        }
        std::cout << "\nZone:";
        for (int i = 0; i < 4; ++i) {
            std::cout << " [" << i << "] " << winnerPerZoneCount[i] << ",";
        }
        std::cout << "\n";

        return populationFitness;  // NEAT训练函数的格式
    };

    while (!WindowShouldClose()) {  // 判断窗口是否关闭
        while (!isStart) {  // 如果没开始训练，则不更新和绘制个体
            keyControl();

            if (WindowShouldClose()) {
                canExit = true;
                break;
            }

            BeginDrawing();
            ClearBackground(BLACK);

            for (int i = 0; i < 4; ++i) {
                DrawCircleV(initPosistion[i], 10.f, GRAY);  // 绘制出生点
                DrawCircleV(targetPosition[i], 10.f, RED);  // 绘制目标点
                for (auto &w: walls[i]) {
                    w.draw();
                }
            }

            DrawFPS(screenWidth / 2 + 10, 10);
            EndDrawing();
        }

        stepCount = 0;  // 重置训练迭代计数器
        auto best = sneat.TrainByInteractive(interactiveFunc, isBreakFunc);  // NEAT训练函数，开始训练

        if (canExit) {
            break;
        }

        isStart = false;
        printf("Pause\n");  // 如果训练循环终止，则重新开始训练
    }

    CloseWindow();   // 关闭窗口

    return 0;
}