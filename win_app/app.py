from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn import metrics

plt.rcParams['axes.unicode_minus'] = False
ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(ROOT / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


st.set_page_config(page_title='Wine selection', layout='wide')
st.title('Wine selection')

model = load_model()
test = load_test()
config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
features = list(model.feature_names_in_)

threshold = st.sidebar.slider(
    'Threshold', 0.0, 1.0, float(config['threshold']), step=0.001, format='%.3f'
)
single, evaluation, batch, coefficients = st.tabs(
    ['Single wine', 'Model metrics', 'Batch prediction', 'Coefficients']
)

with single:
    values = {}
    columns = st.columns(3)

    for i, name in enumerate(features):
        with columns[i % 3]:
            if name == 'is_red':
                values[name] = int(st.toggle('Red wine', help='Off for white wine'))
            else:
                values[name] = st.number_input(
                    name.replace('_', ' '),
                    min_value=float(test[name].min()),
                    max_value=float(test[name].max()),
                    value=float(test[name].median()),
                    format='%.5f'
                )

    X = pd.DataFrame([values], columns=features)
    probability = model.predict_proba(X)[0, 1]
    st.metric('Probability of quality >= 7', f'{probability:.1%}')
    st.write('Recommend' if probability >= threshold else 'Skip')

with evaluation:
    y = test['target']
    scores = test_proba(model, test[features])
    labels = (scores >= threshold).astype(int)

    precision = metrics.precision_score(y, labels, zero_division=0)
    recall = metrics.recall_score(y, labels, zero_division=0)
    f1 = metrics.f1_score(y, labels, zero_division=0)

    for column, name, value in zip(
        st.columns(3), ['Precision', 'Recall', 'F1'], [precision, recall, f1]
    ):
        column.metric(name, f'{value:.3f}')

    cm = metrics.confusion_matrix(y, labels, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    metrics.ConfusionMatrixDisplay(cm).plot(ax=axes[0], colorbar=False)
    axes[0].set_title('Confusion matrix')

    metrics.RocCurveDisplay.from_predictions(y, scores, ax=axes[1])
    axes[1].scatter(fp / (fp + tn), recall, label='Current threshold')
    axes[1].legend()

    metrics.PrecisionRecallDisplay.from_predictions(y, scores, ax=axes[2])
    axes[2].scatter(recall, precision, label='Current threshold')
    axes[2].legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with batch:
    uploaded = st.file_uploader('Upload wine CSV', type='csv')
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            missing = [name for name in features if name not in data]

            if missing:
                st.error('Missing columns: ' + ', '.join(missing))
            else:
                probabilities = model.predict_proba(data[features])[:, 1]
                result = data.assign(
                    probability_good=probabilities,
                    recommend=(probabilities >= threshold).astype(int)
                )
                st.dataframe(result)
                st.download_button(
                    'Download predictions',
                    result.to_csv(index=False).encode('utf-8'),
                    file_name='predictions.csv',
                    mime='text/csv'
                )
        except (ValueError, UnicodeError) as error:
            st.error(str(error))

with coefficients:
    coefs = pd.Series(model.steps[-1][1].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    coefs.plot.barh(ax=ax)
    ax.set_xlabel('Coefficient for standardized feature')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write('Strongest positive: ' + ', '.join(coefs.nlargest(3).index))
    st.write('Strongest negative: ' + ', '.join(coefs.nsmallest(3).index))
