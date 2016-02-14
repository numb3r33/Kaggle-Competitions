from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from features import FeatureTransformer

def build_random_forest_model():
	"""
	Creates a pipeline consisting of feature transformer and 
	a regressor.
	"""

	ft = FeatureTransformer()

	reg = RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=5, n_jobs=-1)
	pipeline = Pipeline([('ft', ft), ('reg', reg)])

	return pipeline

