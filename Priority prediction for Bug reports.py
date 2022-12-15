import sys, os
ON_COLAB = 'google.colab' in sys.modules

if ON_COLAB:
    GIT_ROOT = 'https://github.com/blueprints-for-text-analytics-python/blueprints-text/raw/master'
    os.system(f'wget {GIT_ROOT}/ch06/setup.py')

%run -i setup.py

%run "$BASE_DIR/settings.py"

%reload_ext autoreload
%autoreload 2
%config InlineBackend.figure_format = 'png'

# to print output of all statements and not just the last
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# otherwise text between $ signs will be interpreted as formula and printed in italic
pd.set_option('display.html.use_mathjax', False)

# path to import blueprints packages
sys.path.append(BASE_DIR + '/packages')

# Cleaning up the data to remove special characters 
import html 
import re

# tags like 
RE_TAG = re.compile(r'<[^<>]*>')
# text or code in brackets like [0]
RE_BRACKET = re.compile('\[[^\[\]]*\]')
# text or code in brackets like (0)
RE_BRACKET_1 = re.compile('\([^)]*\)')
# specials that are not part of words; matches # but not #cool
RE_SPECIAL = re.compile(r'(?:^|\s)[&#<>{}\[\]+]+(?:\s|$)')
# standalone sequences of hyphens like --- or ==
RE_HYPHEN_SEQ = re.compile(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)')
# sequences of white spaces
RE_MULTI_SPACE = re.compile('\s+')

### cleaner clean function
def clean(text):
    try:
        text = html.unescape(text)
    except AttributeError:
        print("Attribute error: ignored")
    text = RE_TAG.sub(' ', text)
    text = RE_BRACKET.sub(' ', text)
    text = RE_BRACKET_1.sub(' ', text)
    text = RE_SPECIAL.sub(' ', text)
    text = RE_HYPHEN_SEQ.sub(' ', text)
    text = RE_MULTI_SPACE.sub(' ', text)
    text = text.lower()
    return text.strip()
    
    file = "eclipse_jdt.csv"
file = f"{BASE_DIR}/data/jdt-bugs-dataset/eclipse_jdt.csv.gz" ### real location
df = pd.read_csv(file)

df = df[['Title','Description','Priority']]
df = df.dropna()
df['text'] = df['Title'] + ' ' + df['Description']
df = df.drop(columns=['Title','Description'])

# Data Preparation- Clean and remove short bug report
df['text'] = df['text'].apply(clean)
df = df[df['text'].str.len() > 50]

# Filter bug reports with priority P3 and sample 5000 rows from it
df_sampleP3 = df[df['Priority'] == 'P3'].sample(n=5000)

# Create a separate dataframe containing all other bug reports
df_sampleRest = df[df['Priority'] != 'P3']

# Concatenate the two dataframes to create the new balanced bug reports dataset
df_balanced = pd.concat([df_sampleRest, df_sampleP3])

# Check the status of the class imbalance
df_balanced['Priority'].value_counts()

#Project Step-3 Lemmtization
import spacy
nlp = spacy.load("en_core_web_sm")

nouns_adjectives_verbs = ["NOUN", "PROPN", "ADJ", "ADV", "VERB"]


for i, row in tqdm(df_balanced.iterrows(), total=len(df_balanced)):
    doc = nlp(str(row["text"])) 
    df_balanced.at[i, "lemma"] = " ".join([token.lemma_ for token in doc if token.pos_ in nouns_adjectives_verbs])
    
df_balanced.sample(5)


#Hyperparameter Tuning with 5-fold cross validation Grid Search on LinearSVC()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


training_pipeline1 = Pipeline(
    steps=[('tfidf', TfidfVectorizer(stop_words="english")),
           ('LinearSVC',LinearSVC(random_state=1))])

grid_param = [{
'tfidf__min_df': [5, 10],
'tfidf__ngram_range': [(1, 3), (1, 6)], ### from upto 3-grams to upto 6-grams
'LinearSVC__penalty': ['l2'], ### square L2 penalty
'LinearSVC__loss': ['hinge'], ### standard loss function
'LinearSVC__max_iter': [10000] ### epochs = 10000--run the entire data for upto 10000 times
}, {
'tfidf__min_df': [5, 10],
'tfidf__ngram_range': [(1, 3), (1, 6)],
'LinearSVC__C': [1, 10], ### C is a Regularization parameter; model strength is inversely proportional to C.
'LinearSVC__tol': [1e-2, 1e-3]  ### model tolerance
}]

#GridSearchCV function looks for best alpha parameter by grid search within the parameters specified in grid_param
#cv = k  for k-fold cross validation
#n_jobs Number of jobs to run in parallel.-1 means using all processors
gridSearchProcessor1 = GridSearchCV(estimator=training_pipeline1,
                                   param_grid=grid_param,
                                   cv=5,n_jobs=-1) 
                                   
                                   
 #Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df_balanced['lemma'],
                                                    df_balanced['Priority'],
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=df_balanced['Priority'])
gridSearchProcessor1.fit(X_train, Y_train)
model_LinearSVC = gridSearchProcessor1.best_estimator_
print(model_LinearSVC)

gridSearchProcessor2.fit(X_train, Y_train)
model_SVC = gridSearchProcessor2.best_estimator_
print(model_SVC)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

#comparision of LinearSVC() and SVC() classifiers trained above

#Comparing accuracy, precision, recall, and F1 scores
#Model Evaluations for Linear SVC
Y_pred1 = model_LinearSVC.predict(X_test)
print('Accuracy Score-LinearSVC :', accuracy_score(Y_test, Y_pred1))
print(classification_report(Y_test, Y_pred1))

#Model Evaluations for SVC
Y_pred2 = model_SVC.predict(X_test)
print('Accuracy Score-SVC :', accuracy_score(Y_test, Y_pred2))
print(classification_report(Y_test, Y_pred2))


#Run cross validations for classifiers over the balanced dataset.
#to check model stability
from sklearn.model_selection import cross_val_score

scores_LinearSVC = cross_val_score(estimator=model_LinearSVC,
                         X=df_balanced['lemma'],
                         y=df_balanced['Priority'],
                         cv=5)

scores_SVC = cross_val_score(estimator=model_SVC,
                         X=df_balanced['lemma'],
                         y=df_balanced['Priority'],
                         cv=5)

print ("Mean, Standard deviation value of validation scores  for Linear SVC", scores_LinearSVC.mean(), scores_LinearSVC.std())
print ("Mean, Standard deviation value of validation scores  for SVC", scores_SVC.mean(), scores_SVC.std())

#Retain the Best alpha parameter identified by grid search for SVC()
best_svc = gridSearchProcessor2.best_params_

#vectorize with best paramters identified by grid search
tfidf = TfidfVectorizer(min_df=best_svc['tfidf__min_df'],
                        ngram_range=best_svc['tfidf__ngram_range'],
                        stop_words='english')

X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)

#the best hyperparameters except the kernel types to train 
#SVC(kernel=’linear’)
svc_linear_kernel = SVC(C=best_svc['SVC__C'],
                        kernel='linear', 
                        probability=True,
                        random_state=42, 
                        tol=best_svc['SVC__tol']
                        )

#the best hyperparameters except the kernel types to train 
#SVC(kernel=’rbf’)
svc_rbf_kernel = SVC(C=best_svc['SVC__C'], 
                     gamma=best_svc['SVC__gamma'],
                     kernel='rbf', 
                     probability=True,
                     random_state=42, 
                     tol=best_svc['SVC__tol']
                    )
svc_linear_kernel.fit(X_train_tf, Y_train)
svc_rbf_kernel.fit(X_train_tf, Y_train)
#Model Evaluations
##SVC  with linear Kernel
Y_pred_linear = svc_linear_kernel.predict(X_test_tf)
print('Accuracy Score- for SVC with linear kernel :', accuracy_score(Y_test, Y_pred_linear))
print(classification_report(Y_test, Y_pred_linear))


##SVC  with rbf Kernel
Y_pred_rbf = svc_rbf_kernel.predict(X_test_tf)
print('Accuracy Score- for SVC with rbf kernel :', accuracy_score(Y_test, Y_pred_rbf))
print(classification_report(Y_test, Y_pred_rbf))

import eli5
eli5.show_weights(model_LinearSVC, top=5, vec=tfidf, target_names=class_names)

import eli5
eli5.show_weights(model_SVC, top=5, vec=tfidf, target_names=class_names)

from lime.lime_text import LimeTextExplainer
from lime import submodular_pick

explainer = LimeTextExplainer(class_names=class_names)
lsm = submodular_pick.SubmodularPick(explainer, er["text"], model_SVC.predict_proba, 
                                        sample_size=100,
                                        num_features=5,
                                        num_exps_desired=5)


lsm.explanations[0].show_in_notebook()
lsm.explanations[1].show_in_notebook()
lsm.explanations[2].show_in_notebook()
lsm.explanations[3].show_in_notebook()
lsm.explanations[4].show_in_notebook()
lsm.explanations[5].show_in_notebook()
