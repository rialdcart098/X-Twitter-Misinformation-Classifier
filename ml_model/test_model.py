import dill as pickle
import numpy as np
def main():
    tweet = "I've said this before, but it really is incredibly the way in which Afghanistan has completely crowded out the things that are *actually* affecting ordinary Americans in national media coverage, from COVID to the eviction moratorium to climate change."
    with open('model.pkl', 'rb') as fin:
        model = pickle.load(fin)
    with open('vectorizer.pkl', 'rb') as fin:
        vectorizer = pickle.load(fin)
    processed_tweet = vectorizer.transform([tweet])
    certainties = model.predict_proba(processed_tweet)
    prediction = np.argmax(certainties, axis=1)
    confidence = np.round(certainties[0, prediction[0]] * 100, 2)
    print(bool(prediction[0]), float(confidence))
if __name__ == '__main__':
    main()