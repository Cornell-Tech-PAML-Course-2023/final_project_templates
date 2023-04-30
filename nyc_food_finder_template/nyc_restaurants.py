import streamlit as st        
from st_clickable_images import clickable_images
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This final project proposal template provides an example of restarant recommendation application to helps visitors find good restaruants in NYC. 
# The layout has one-page (instead of sidebar/tabs). 
# The websites sections include 
# - Explore NYC Restureants:
# - Recommendation Analysis: Shoes classification performance on dummy data including precision, recall, TP, TN, FP, and FN

def apply_threshold(probabilities, threshold):
    # +1 if >= threshold and -1 otherwise.
    return np.array([1 if p[1] >= threshold else 0 for p in probabilities])

tab1, tab2 = st.tabs(["Cornell Tech Restaurant Finder", "Restaurant Recommendation Analysis"])

with tab1:
    header_col_front_1, header_col_front_2 = st.columns(2)

with header_col_front_1:
    st.image('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAW4AAACJCAMAAAA7SiIzAAAAh1BMVEX///8AAADGxsbk5OT6+vp8fHz19fUxMTHh4eFzc3PPz8+oqKiamprw8PC1tbVmZmZfX1/r6+uCgoIjIyPb29uioqLOzs5LS0u6uroNDQ3BwcEcHBzW1tasrKwqKiqVlZU7OztNTU1DQ0MVFRVZWVmNjY15eXk9PT2WlpYgICBcXFxlZWVubm42bvBlAAATXklEQVR4nO2diZaquBaGEwiTAgoyC8qo5fD+z3czQlS0qk6fdbuszr/O6lYCCB9hZ2dnJwWAkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSktJ/UihK/P3gZmXwWGYXByuH23ZYyIXa1ZF0CKubQ6v1arW6LOXTeGe8abXOpvJ7Xa60zLmQz3v0cCGBQUvcf3y3/0TIXmiFn2mLyPzjc2TOBjKle/u2SN/voFBaTT+xhPfqq4mQQbecSulEAduLwQofjqaiZR39uH68G5tdSf/Ht/nPhMxyuLS1VdcWVX3sQ818rBWfyW7kW95Wclm2veHRRKLgETeEF12UcpyNdDEcdyyXz+JmT/4yg9tiD+LbN/gXhMplD2tnqDReq01dq/aHFqZu8r1q7p/ubtoZj0fDAxGNF83hhrkwKAKn9OK/M25zeYZtJYwpsnEtCuj1BaW/hv1+xgQ/U/Z41wdR9vFYtk1Y0SxuuPZYqcB5Gt+GN8YdGHkb8xszjRrEeQLMKyyTK32bzWXfrewvGpVyO3PbvFKW4ntuHUcLvmOPkuPOme6PHHGuRmpzuPM70bIfhtseoJONMJEBgZ8XwEvhUEEbmC553bUDNBZfORl+Skz1vqriXiCllRKJr05h47dGfBvogRz3PqPyhdXZsocx1d5Y/NIM7s7NblTQsp+Fu9q0CYZd9fx69hAUuQ+CDTyHkNxVDMwEe3ZrzuW1RAV2KCavkpFq3Ki77NkGDvva0W8cdyJOlPAjmaMnGQv+Fs7h3sqey6ifhLu0LFYHNMj+D2LoLfIK6DCG3Y7Qj0DUEVxl0/mfno9zWYnvPodEbtZlH0NR5vH6Tc/KcWvg7shePi3dwF/EOdyzL+APwr2EuBW0SYXRa+5/VnAI8V2XULdgA8CxD4CWL4HvIzOG+8+cFHYH27FFQ8ZYa4Wd8cadK6nqP+C2a7rBol/kppA/8/fDjc7YSIMgp/VtyAOTWAAUbne4gSo6PcYuhQ33xNlY+NSd00+p/fqUjMF1uresIyJnCVJaVk8762xvg3x+wO2t6QZmamTcO+aNvx3uaNcTeOgAyQ1onTXZACrTLzGFgtRCFzr4HS6x3zJZ1zkhxkByj5EdEeEHGTBn5Crtnk9bHnDr6RPcnM+74S43uMJWMfHdrgDpYJdWrOpGSZEVSURtZED6ODGrr9o2JJ+LF+eMZAZ34nU5lDYxEmdiXh5w87aypl9u/Wp6BW/WVCa4DpoD3ON7HeCqCwHG69lVg3vvxzZN26NlNS4167i5NHCh3lnb3sNNWPX8pAVjsJwrs1nZXtrEzHNDXq573AEr4+4Qd/RY+AMeSbWYw62Z3iTB90fgXpA33ofnYm0CE9c5fMuJa20url/qlLFXZrFh7fYaAp6PqB8RJmFAkGZPz8q7lLMPxH4wNKClW1IJd+YFRFHW8prMGl2Gc7fnvIm1n+vmHNNJrfihn4A76Ogl93l3ACYosNHTepgm+t1FBaUDU2Y9QujwquY/t98c96zDaD8amvQBd91QVrWwGw7iP86MhTAqyRc68Qb/kR+AO8gH4JHm61gn9RXf0uK0e+bkVfWuQMDcwYiEoPVmgZvOWSMJRmf5FW7Z0DDcLbEN8zGTI3eExqaQV/rOfCfc5mWFQN/YxKbAugSBAV396d5edTJsoK8I5LKGrQmG45OY1d/GLR7riLvg5mR4J9zuBpvnLN+UpPPu4aq9fu1Oe86pAKgysXMCK9LFPBrzEasv4H5tTG7Uj72lydHj3aY88d4Gd4JrzfKSuV1Oq9p+3pO4UbEZEDHN1vZI4qnBdtbX+y7ux6ZSUi55nBNuxMeJ6pvTcdyb3aSTCPH827jNdg88/lLiNifcPPc0JkXplbwKcde6xJD4Ul9c0jc9kyPdIjuC/fV6XXPeknmTujEinu484u6qqJwk7N2/jfujxv8p03xw+9BETvvcassKVj12CR2KGVf063VuJ87ild8txxXZ/fYEjOR3I15XL9OOEm50azh+fDcn6mhfN7jQimZYtvbhfkGxvboic006hb5ToWDW+y5lBnfS2QvlSJvYlhVpB+RuTsIitd3UxZQ76froJT7g/omd+BW+4ZL0Ww4QgQ/cXB7gl7TwUoMOqRGfwAf7dMZxvBkevy/jdVnaxGImdGztplfJzclhbI9vcPryVf103DruqDlwSyyICfw8IjHSEiCE/90IBLZQgK3HBS6AbuE2cDhEMGodFFhz1ZsxcKSIINsSTuHtyafRxrI73NzuTF36W5yXN8LtONgFHLb9EGNfehPTkDRxuMz9wLVkQD5Ooo0nt7TCuLGXXoIwX0KtxEe49czZWX21puaAj4IVk9GdnlIoEbuNmfDq3d7uKXBG1tvgLnMbu7t9d4K4q7ImZpPjDrY5yy6BfBhtgP15vV73a2obSO3GPuMZ2TW0qD8cWDOxwZVUYYnEGBhpYPlowthFEl0W6vPf4haHiep9hzN+gvtVU/nYUfh/4HZxjSmv+DIDBIoTqYUC94Y3cCHHHWLjbBf4s4TbTCsQVAFJS4lAOOOcaIICPQcS2GiXQ7RxR3JSfBpexoz5LW7E25OG18n72juZkxvcCTJvRMsY7rX3UMRxn+cO+kvyjhnFsMZ3abKgmcC966k52bdnjrs1sx3sFzJu3N2htcR34BnYXfR4fjG+fok1rXJ4BkPOYlpjUs9q2IdiRx7vugvAavxIbnnucQvjfos7J177pBWrDgz3bnVTRrrRHPfm9qDDJ+NV31K5CajbDLQEv9zgBnfHctVEOt1gXeFBa6Er4wbrPUDDDm7IUHs9050pxhwRKVtE+NqSzb0vu493n3n19iSckm0ezclX03ruVY647zTfAPyhriEI4kNMjJzXsksdjQl33w6j7bYqlmcm4y6twNysK0qhamYiJ/HjHYx2024fygz+8t7jFhkUvoRTIuGJrudfx737m7g3FWj4hS9OrM0SuE8ctyFst8VaHht+AOIr8MuoMzZ0v9ijEs4Fqh54N1P4kA9ATgqFqXwYPOPGZseu5R43CWX+fNy2pdvYKdNT3GD2fBiL4wZNSvGWGx5KK3noI6FjgyPujCd6aDDwUg3MaNHIl5/fZlbv5T5hP/k2z8YqWZdpxtHbvwHuZYNvIyF2GSDRDAncix1cIuTmtXCaXTKkFgyM+ojbJkl9kR+fMRrj8PALRGa1Hq/euPcW9Zib5dNBSu9+xC3SUujlcJzyKJKZTk/jJe58tojgnn0Q3V/EbeAOPLSGoTuArOYxPYEb6FfYpPAwvfqL2jJqS/Sr+WWgawXcE6aFzxG3YF6mXrjGeW0UwYy58fSoqBL7tsjUqUx5t2kT/4xuT0Pk3ex7J3ZbcyIZpmj+oO8ntD8TWuOaGq13J9y9MUSsaMSN3YrmfNtTcBsR/hhxA/cMitAPQNwArf571/YLFdQ0jycqEUC9iJJi3GOf+zm9y4hbq/nJsKWxHj1vpVFljt9aghRhEyzqMca9sz5VPuL2xHgtPstsmEqJS8vxP2yLqwE7UmKjyJr8RPnYhLSs9QuGJWjnx9CUqIodG/WKU9z7Exsx7qq8lQb3d1uifqzdgJmhpMGuTf/vzpL74cpqkMGlGTQGyCyxkca7b2U+jjauJtxXF/hHi067u+zv91OalKUkhtQcsXvry7jvG7wZ3FNTCQ4DqNoL3Lk6cL4yo+E/q6whY2ZWg7sLfi02yp4JlV4mMPbuApES7nAApmfaBzgAQ+F+IVy7hfyxh4Jxi56Nc988bqGgflu7qfRs/Kg0p2zqBWIzzoVxC6arXexnRbIoYJgkWuFX4Vg0dXPA5WM8y1XZ7hcqrOmj7AiK3s219aKiyGxsuxEZa0DaWCThbibvTyKv9CBt9P5AIuNmH4KiST1iTwYPVsHxeKw9DWa8GZVwt1PfJr33uwttVgUxV8F8mdyT1bXMj6tMu2lMbLpb4sm70ZPOJnL9IE19GxBtxcWOuAsI0yAg6V4YN43skKFHPso74UbHyW+s77IB0bNOEnlC/nzRhDYKUx6+Sw3JWeK59FKk7iF8+DOlW2O4z2zGOjpw3OVqk7JyTziCGhShlX7sVWrWNMW4votWPsVNHssnuMcJx0zhWHdZr3cjRV/j98CNesYYJSU4jLFqd/RMsO0u4mW8wLjN5XIZY9stdmo3ok7HDYh4ELus797nf4A7uh/oGQeBGO6dhPtNajfrliRhk69JR57LHyeTrVpquz886NN5YsHYVCJrDExht3tokEvmUFX13fn/HLf+OLTS82f5vrjdnsx7gqd+p3uduH6N22dkrtsHY2Kb/AOfJYO52Ga/cmgOyHCfafLHuFE3U3JlP/m+uBf47Q8zd5fkFaiFVxG0bER9cd6Otps3gRpsme12xwl6hYUWuEGrKhig9X3ePFrxxaR43k3Kv16Jjee4m4O86pRz9dj5BeLKHxeK4FfxvrjBidgN1No6wv1KYXiNnF44dkNSb3nojz7GHbRY2BFkmIN+DGP1MUCRYQVhY0bdgysmEjp53sJSbCBl/t02qUhkRDQljcaLZJUNu7z3xc2H370yAMFWmOySTlMFpt23Xpg2KcHtHcLQMDVY0HfdJ6sRUAUkZ5Z8CDXcM306+hMJtJI47rmpDTxP1hKPb8FXl6GI3xh3kdLkyGabamBfi60pd6pXqYc805xG6RPeVG6m5EgHJAansnoe7f4m7ilPlktejOONcQckSl3B2kgh0Hfi9iJ4olhXp8EdnL7dwLRt1oMbH1jMxGEzDLCSrQ4sx09dk4Rpn+cIfA+32TDbMdkm3eq2221O10R4Y9zUmWgbDxuQAsS1cG33rB6RvNNt3V9qaMWY+2W9XhPMFV/diGQ9GABtznVNUvvi9fOR5G/i5rmoUtA3IeGAgj7Pd8ZNQlPpgSDEVTsX1sA70heZTFjQA8/0aaIkMk2P1LdoM0620TY67VQXh9NMwETS93DzOZJPJmu+M26wHnDDlNKVeHAVF+Yg6OSFMwooRfqiaWW6gO00dCRZzN+9+JU/wv0kmstxSyN8b4Rb2wUgbo8fAUAe+KilqQRT9SmkOy/rcaKF14cA+diuJB9L01y/GoV/gfu6jCcx9jczgB/EcG/D8aDl6n1wgx2+K2IkvIOFHUOxMCKu8t1YvyXcyWZcO5FkFoIrPLGv2unVWo4vcN+IvSHezN6TniZmvAVuraPRD9sizSNqhc9BlmeN+ecRN1pKc1f2VoCRXbbWhZxg82IVGYV7EnJIdmBU5xnwMqDXYwOlWSK5XeAmU2TGDOwPMjekgBZckecUty8nsfwR7icP8L1xA5v42zEMQJRDG3htL2J9QcNzpThuzL8TwMxwEwDc9deucNnFoOyeLWnCpHBPWmKrUEEthhsSA/LWU7qg28FB57h17PBdRIl9bnQQbfeIPJOVZzbzmd2jvop7S0t+szHBWjtkjVc6o3GogLmfZlVHW3j0gYY3LI9wWqXRzw8mjWy4ZMEkDxjtJ2sev8B9+dhPYs7I1zyTaslVXd8Ld1CTVdbIWqwunTSd1N1CWHDXgtYAz9i1XomuZLlmM1aL+sBnVG8/SzT+I7/7ybD+W/vdVAveqSkgbE1XB14MnTGAwuJFvYhll5gxq8umjRxSBRfz68PI+iPc4S/sVTJltD8ZbXbHvUHDf3a4bcQfO9AdeMqYHUHZ+uRIEzvRiix+/Hl2yTdjJmws5/prcbPV0tKNDXv8DpsxvnLvo4dpvKBzZkhPxosWxRo2e51keiwmEnb3hdSpb+Jmy/ZIA81mX5NEFzqh5TfgBnG3AIsyonG4gqTy4HtMqgs8tSsjxDo0HWyWJJlGH2ppUYFo8ySSdKPv4UZ8rG1KOeHLbdKx0F+BGyxzupquha3G7hgTw0kHrsxFvA8P4RCPf1QkhpfAgD4aaHgQDl+Z/vTN4QXOb1wOKeil438Hbmy/sbHel+TDEjTQA8trdZNCFrA79I5bgHaDRm6v+MKqbETfxJ3wv9jwwReuFxk+9OsvwY09DGobzR53yLc7k+Ybe2B/CDnmA18NKobLoAtBSULdLyMlk17gPvjVjUibjMS813PmgSAWGT6sL/VbcAPvSpa+WNRwtcNkojoNsL9y7Gq2DIO9g2wVNvI3KkhKRDn19z/TV3uVwmBHcyW8L/VrcAMUd9intt2G/CEXDebYJzRrw6stgjWCIWkhAx2bELyX6W7Dl2EpWd/FPbdQRM5x/h7cZA4xWUbDJK1gDKMIv8pWCA40q2QBGwgXQY8tjk1S8dvXUanb034X9wxvkTP6m3CTGSS5T//oggPzwQTRLtR39DV2obOvNyYZcfGKtv5aG8n1fdxAu1kfEE6rSfwu3JhlXQ8R9q+TGPt6i7zrWCf93JmgJOtG6XG7q77xd7jAH+EGujslCubSistsXart+yUcPxPS1lbrsiUYvKzKGNqE1vnAb6w0+4qvLUunf37yegNk4czpKiW+mYnRtPWxvWZyNpxPzxVKrbRGj1z9zVWj/s/Sl6uTFfpaJHWk7cTft9s+fuPb+rky7cVQ59v2fHUOhuFcL+0mt0KZv9JfFyp93Ik3wn1cLb5rQZSUlJSUlJSUlJSUlJSUlJSUlJSUlJSUlJSUlJSUlJSUlJSUlJT+I/ofk+ZkI1t1ab4AAAAASUVORK5CYII=')

with header_col_front_2:
    st.header("Cornell Tech Restaurant Finder") 

st.markdown('### A fast, personalized recommendation to search for the best restaurants for you!')

st.write('At Cornell Tech Restaurant Finder, we provide personalized restaurant recommendation based on your preferences. Using machine learning algorithms, we find the right restaurant for you! Get prepared for delicious eats and foody experiences in New York City! ')

st.markdown('### What borough are you dining in today?')

categories = []

boroughs = ['Manhattan', 'Queens', 'The Bronx', 'Staten Island']

# Shop by category
borough_select = clickable_images(
    [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Chinatown_manhattan_2009.JPG/440px-Chinatown_manhattan_2009.JPG",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Greenpoint_Houses.JPG/440px-Greenpoint_Houses.JPG",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Unisphere_in_summer.jpg/440px-Unisphere_in_summer.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Melrosebx1.JPG/440px-Melrosebx1.JPG",
    ],
    titles=boroughs,
    div_style={"display": "flex", "justify-content": "left", "flex-wrap": "wrap"},
    img_style={"margin": "10px", "height": "110px"},
)
st.markdown(f"Boroughs {boroughs[borough_select]} Selected" if borough_select > -1 else "")

borough_col1, borough_col2, borough_col3, borough_col4 = st.columns([1,1,1,1])

with(borough_col1):
    st.write(boroughs[0])
with(borough_col2):
    st.write(boroughs[1])
with(borough_col3):
    st.write(boroughs[2])
with(borough_col4):
    st.write(boroughs[3])

df = pd.read_csv('food_order.csv')

# Show map of California
st.markdown('## What is your desired location?')
if (('latitude' in df.columns) and ('longitude' in df.columns)):
    df = np.dropnan(df)
    st.map(df)
else:
    st.write('No longitude and latitude data to display')

st.markdown('### Select a Recommendation System')

method_col1, method_col2, method_col3 = st.columns([1,1,1])

method_select=-1
with(method_col1):
    st.write("Recommendation 1")
    if st.button('Select 1'):
        st.write('Recommendation 1 Selected')
        method_select=1
with(method_col2):
    st.write("Recommendation 2")
    if st.button('Select 2'):
        st.write('Recommendation 2 Selected')
        method_select=2
with(method_col3):
    st.write("Recommendation 3")
    if st.button('Select 3'):
        st.write('Recommendation 3 Selected')
        method_select=3

st.markdown('### Your Recommendations')

with tab2:
header_col_rec_1, header_col_rec_2 = st.columns(2)

with header_col_rec_1:
    st.image('data:image/png;base64,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')

with header_col_rec_2:
    st.header("Cornell Tech Restaurant Finder") 


st.write('## Recommendation Evaluation Performance')

###############
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)
pipeline = make_pipeline(StandardScaler(),
    RandomForestClassifier(n_estimators=10, max_features=5, max_depth=2, random_state=1))
pipeline.fit(X_train, y_train)
# Get the predictions
y_pred = pipeline.predict(X_test)
# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

st.write('### Classification Metrics')

st.write('At Cornell Tech Restaurant Finder, we are transparent about the machine learning model performance of the recommendation systems used on our website. We have provided results including evaluation metrics: precision, recall, true positives, true negatives, false positives, and false negatives. We share the results on a test dataset.')

true_neg, false_pos, false_neg, true_pos = conf_matrix.ravel()

precision = true_pos/(true_pos+false_pos)
recall = true_pos / (true_pos + false_neg)

st.write("Precision: %s" % precision)
st.write("Recall: %s" % recall)

st.write('There are {} false positives'.format(false_pos))
st.write('There are {} false negatives'.format(false_neg))
st.write('There are {} true positives'.format(true_pos))
st.write('There are {} true negatives'.format(true_neg))

st.write('### Classification Confusion Matrix')

# Add intro text

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
st.pyplot(fig)

st.write('### Classification Precision-Recall')

# Add intro text

precision_all = []
recall_all = []
threshold_values = np.linspace(0.5, 1, num=100)

from sklearn.metrics import recall_score, precision_score

probabilities = pipeline.predict_proba(X_test)

for threshold in threshold_values:
    predictions = apply_threshold(probabilities, threshold)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    
    precision_all.append(precision)
    recall_all.append(recall)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
plt.rcParams['figure.figsize'] = 7, 5
plt.locator_params(axis = 'x', nbins = 5)
plt.plot(recall_all, precision_all, 'b-', linewidth=4.0, color = '#B0017F')
plt.title('Precision recall curve (all)')
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.rcParams.update({'font.size': 16})

st.pyplot(fig)