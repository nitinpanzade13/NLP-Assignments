import nltk
from nltk.corpus import conll2002
from nltk.tag import ClassifierBasedTagger
from nltk.classify import MaxentClassifier

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Downloads
nltk.download("conll2002")
nltk.download("punkt")

# -----------------------------
# Load dataset
# conll2002 gives: (word, pos, iob-tag)
# -----------------------------
train_sents_raw = conll2002.iob_sents("esp.train")
test_sents_raw  = conll2002.iob_sents("esp.testb")

print(f"Train sentences: {len(train_sents_raw)}")
print(f"Test sentences : {len(test_sents_raw)}")

# -----------------------------
# Convert (word,pos,tag) -> (word,tag)
# -----------------------------
def convert_format(sents):
    converted = []
    for sent in sents:
        converted.append([(w, ner) for (w, pos, ner) in sent])
    return converted

train_sents = convert_format(train_sents_raw)
test_sents  = convert_format(test_sents_raw)

# -----------------------------
# Feature extractor (FIXED)
# Must accept (tokens, index, history)
# -----------------------------
def word_features(tokens, index, history):
    word = tokens[index]

    features = {
        "word": word,
        "lower": word.lower(),
        "is_title": word.istitle(),
        "is_upper": word.isupper(),
        "is_digit": word.isdigit(),
    }

    # Previous token
    if index > 0:
        prev_word = tokens[index - 1]
        features.update({
            "prev_word": prev_word,
            "prev_lower": prev_word.lower(),
        })
    else:
        features["BOS"] = True

    # Next token
    if index < len(tokens) - 1:
        next_word = tokens[index + 1]
        features.update({
            "next_word": next_word,
            "next_lower": next_word.lower(),
        })
    else:
        features["EOS"] = True

    # Previous predicted tag history
    if history:
        features["prev_tag"] = history[-1]
    else:
        features["prev_tag"] = "<START>"

    return features

# -----------------------------
# Maxent builder (training iterations)
# -----------------------------
def train_maxent_classifier(train_feats):
    return MaxentClassifier.train(train_feats, max_iter=10)

# -----------------------------
# Train NER tagger
# -----------------------------
tagger = ClassifierBasedTagger(
    train=train_sents,
    feature_detector=word_features,
    classifier_builder=train_maxent_classifier
)

print("\nNER model training completed ✅")

# -----------------------------
# Evaluate on test data
# -----------------------------
y_true = []
y_pred = []

for sent in test_sents:
    words = [w for (w, t) in sent]
    true_tags = [t for (w, t) in sent]
    pred_tags = [tag for (_, tag) in tagger.tag(words)]

    y_true.extend(true_tags)
    y_pred.extend(pred_tags)

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted"
)

print("\n=========== EVALUATION METRICS ===========")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print("=========================================\n")

print("DETAILED CLASSIFICATION REPORT:\n")
print(classification_report(y_true, y_pred))

# -----------------------------
# Test on real-world text
# -----------------------------
real_text = "Barack Obama visited Microsoft headquarters in Washington to meet Satya Nadella."

from nltk.tokenize import TreebankWordTokenizer
tokens = TreebankWordTokenizer().tokenize(real_text)

predicted = tagger.tag(tokens)

print("\nREAL WORLD TEXT:")
print(real_text)

print("\nNER OUTPUT:")
print(predicted)
