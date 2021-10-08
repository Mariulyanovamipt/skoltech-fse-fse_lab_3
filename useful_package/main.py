import sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics import mean_squared_error as mse_score

from module_a import polynom3 
from module_b import hyperbola 
x = np.arange(1, 100, 1).reshape(1, -1)
y_real_poly = polynom3(x) 
print(y_real_poly)
y_real_hyp=hyperbola(x)

model1 = RF() 
model1.fit(x, y_real_poly) 

model2 = RF()
model2.fit(x, y_real_hyp)

y_pred_poly = model1.predict(x)
print('MSE score polynom', mse_score(y_real_poly, y_pred_poly))


y_pred_hyp = model2.predict(x)
print('MSE score polynom', mse_score(y_real_hyp, y_pred_hyp))
