import extract_feature as EF
import xgb_useing as model
print('提取特征')
EF.extract('windows')
print('模型训练')
model.run_model('windows')