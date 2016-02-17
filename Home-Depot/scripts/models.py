from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from features import FeatureTransformer

def build_random_forest_model():
	"""
	Creates a pipeline consisting of feature transformer and 
	a regressor.
	"""

	ft = FeatureTransformer()

	reg = RandomForestRegressor(n_estimators=325, max_depth=8, n_jobs=-1)
	pipeline = Pipeline([('ft', ft), ('reg', reg)])

	return pipeline


def build_linear_model():
	"""
	Creates a pipeline consisting of feature transformer and passive
	aggressive regressor
	"""

	ft = FeatureTransformer()
	scaler = StandardScaler()

	reg = PassiveAggressiveRegressor(C=0.1)
	pipeline = Pipeline([('ft', ft), ('scaler', scaler), ('reg', reg)])

	return pipeline