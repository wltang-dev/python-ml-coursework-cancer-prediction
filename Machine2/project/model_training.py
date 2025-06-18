# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVR
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from sklearn.neural_network import MLPRegressor
# from config import RANDOM_FOREST_PARAMS, GRADIENT_BOOSTING_PARAMS, RANDOM_STATE
#
# # 随机森林回归
# def train_random_forest(X_train, y_train):
#     grid_search = GridSearchCV(
#         RandomForestRegressor(random_state=RANDOM_STATE),
#         RANDOM_FOREST_PARAMS,
#         cv=5,
#         scoring='r2',
#     )
#     grid_search.fit(X_train, y_train)
#     print("Random Forest Best Parameters:", grid_search.best_params_)
#     return grid_search.best_estimator_
#
# # 梯度提升回归
# def train_gradient_boosting(X_train, y_train):
#     grid_search = GridSearchCV(
#         GradientBoostingRegressor(random_state=RANDOM_STATE),
#         GRADIENT_BOOSTING_PARAMS,
#         cv=5,
#         scoring='r2',
#     )
#     grid_search.fit(X_train, y_train)
#     print("Gradient Boosting Best Parameters:", grid_search.best_params_)
#     return grid_search.best_estimator_
#
# # 线性回归
# def train_linear_regression(X_train, y_train):
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     print("Linear Regression does not require parameter tuning.")
#     return model
#
# # 支持向量机
# def train_svm(X_train, y_train):
#     SVM_PARAMS = {
#         'C': [0.1, 1, 10, 100],
#         'gamma': ['scale', 'auto'],
#         'kernel': ['linear', 'rbf']
#     }
#     random_search = RandomizedSearchCV(
#         estimator=SVR(),
#         param_distributions=SVM_PARAMS,
#         n_iter=10,
#         cv=3,
#         scoring='r2',
#         n_jobs=-1,
#         random_state=42
#     )
#     random_search.fit(X_train, y_train)
#     print("SVM Best Parameters:", random_search.best_params_)
#     return random_search.best_estimator_
#
# # XGBoost 回归
# def train_xgboost(X_train, y_train):
#     XGBOOST_PARAMS = {
#         'n_estimators': [50, 100, 200],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'max_depth': [3, 5, 7],
#         'subsample': [0.8, 1.0]
#     }
#     grid_search = GridSearchCV(
#         estimator=XGBRegressor(random_state=RANDOM_STATE),
#         param_grid=XGBOOST_PARAMS,
#         cv=5,
#         scoring='r2',
#         n_jobs=-1
#     )
#     grid_search.fit(X_train, y_train)
#     print("XGBoost Best Parameters:", grid_search.best_params_)
#     return grid_search.best_estimator_
#
# # LightGBM 回归
# def train_lightgbm(X_train, y_train):
#     LIGHTGBM_PARAMS = {
#         'n_estimators': [50, 100, 200],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'num_leaves': [20, 31, 40],
#         'max_depth': [-1, 5, 10]
#     }
#     grid_search = GridSearchCV(
#         estimator=LGBMRegressor(random_state=RANDOM_STATE),
#         param_grid=LIGHTGBM_PARAMS,
#         cv=5,
#         scoring='r2',
#         n_jobs=-1
#     )
#     grid_search.fit(X_train, y_train)
#     print("LightGBM Best Parameters:", grid_search.best_params_)
#     return grid_search.best_estimator_
#
# # def train_mlp(X_train, y_train):
# #     # 定义更广的参数范围
# #     MLP_PARAMS = {
# #         'hidden_layer_sizes': [(64,), (128,), (128, 64), (256, 128), (256, 128, 64)],
# #         'activation': ['relu', 'tanh'],  # 增加 'tanh' 激活函数
# #         'solver': ['adam', 'sgd'],  # 尝试 SGD 和 Adam 两种优化器
# #         'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],  # 扩展学习率范围
# #         'alpha': [0.0001, 0.001, 0.01, 0.1],  # L2 正则化
# #         'max_iter': [500, 1000],  # 增加最大迭代次数，确保收敛
# #         'batch_size': [32, 64, 128]  # 增加批量大小选项
# #     }
# #
# #     # 使用 GridSearchCV 进行网格搜索
# #     grid_search = GridSearchCV(
# #         estimator=MLPRegressor(
# #             random_state=RANDOM_STATE,  # 确保随机性一致
# #             early_stopping=True,  # 启用 Early Stopping
# #             n_iter_no_change=10  # 连续 10 次无提升时停止
# #         ),
# #         param_grid=MLP_PARAMS,
# #         cv=3,  # 3 折交叉验证
# #         scoring='r2',  # 使用 R² 作为评估指标
# #         n_jobs=-1,  # 并行加速
# #         verbose=1  # 输出日志信息
# #     )
# #
# #     # 模型训练
# #     grid_search.fit(X_train, y_train)
# #
# #     # 输出最佳参数
# #     print("MLP Best Parameters:", grid_search.best_params_)
# #
# #     # 返回最佳模型
# #     return grid_search.best_estimator_
# # 简单神经网络（MLP）
# def train_mlp(X_train, y_train):
#     MLP_PARAMS = {
#         'hidden_layer_sizes': [(64,), (128, 64), (128, 64, 32)],
#         'learning_rate_init': [0.001, 0.01, 0.1],
#         'alpha': [0.0001, 0.001, 0.01],
#         'max_iter': [200, 500]
#     }
#     grid_search = GridSearchCV(
#         estimator=MLPRegressor(random_state=RANDOM_STATE),
#         param_grid=MLP_PARAMS,
#         cv=3,
#         scoring='r2',
#         n_jobs=-1
#     )
#     grid_search.fit(X_train, y_train)
#     print("MLP Best Parameters:", grid_search.best_params_)
#     return grid_search.best_estimator_